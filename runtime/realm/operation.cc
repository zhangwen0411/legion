/* Copyright 2015 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// implementation of Operation stuff in Realm

#include "operation.h"

#include "utilities.h"
#include "lowlevel_impl.h"
#include "realm/faults.h"

static inline unsigned long long now(void)
{
  return LegionRuntime::TimeStamp::get_current_time_in_nanos();
}

// import a few things from LegionRuntime::LowLevel
typedef LegionRuntime::LowLevel::AutoHSLLock AutoHSLLock;
typedef LegionRuntime::LowLevel::GenEventImpl GenEventImpl;

namespace Realm {

  LegionRuntime::Logger::Category log_op("realmop");
  LegionRuntime::Logger::Category log_optable("optable");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Operation
  //

  Operation::Operation(Event _finish_event, int _priority, ProfilingRequestSet *_prs)
    : refcount(1), finish_event(_finish_event), priority(_priority), prs(_prs)
  {
    assert(finish_event.exists());

    // initialize profiling stuff
    if(prs)
      pmc.import_requests(*prs);

    status.result = ProfilingMeasurements::OperationStatus::WAITING;
    status.error_code = 0;

    timeline.create_time = now();
    timeline.ready_time = 0;
    timeline.start_time = 0;
    timeline.end_time = 0;

    log_op.info("operation %p created: finish=" IDFMT "/%d",
		this, finish_event.id, finish_event.gen);
  }

  /*virtual*/ Operation::~Operation(void)
  {
    if(prs)
      delete prs;
  }

  // these are called as the operation progresses through its life cycle
  /*virtual*/ bool Operation::mark_ready(void)
  {
    // take a lock here to avoid race with cancellation request
    AutoHSLLock a(mutex);

    switch(status.result) {
    case ProfilingMeasurements::OperationStatus::WAITING:
      {
	status.result = ProfilingMeasurements::OperationStatus::READY;
	timeline.ready_time = now();

	log_op.info("operation %p ready: finish=" IDFMT "/%d",
		    this, finish_event.id, finish_event.gen);
	return true;
      }
      
    case ProfilingMeasurements::OperationStatus::CANCELLED:
      {
	// lost the race with a cancellation request - nothing to do
	return false;
      }

    default:
      {
	assert(0 && "mark_ready() called on an operation that isn't WAITING or CANCELLED");
	return false;
      }
    }
  }
  
  /*virtual*/ bool Operation::mark_started(void)
  {
    // take a lock here to avoid race with cancellation request
    AutoHSLLock a(mutex);

    switch(status.result) {
    case ProfilingMeasurements::OperationStatus::READY:
      {
	status.result = ProfilingMeasurements::OperationStatus::RUNNING;
	timeline.start_time = now();

	log_op.info("operation %p started: finish=" IDFMT "/%d",
		    this, finish_event.id, finish_event.gen);
	return true;
      }
      
    case ProfilingMeasurements::OperationStatus::CANCELLED:
      {
	// lost the race with a cancellation request - nothing to do
	return false;
      }

    default:
      {
	assert(0 && "mark_started() called on an operation that isn't READY or CANCELLED");
	return false;
      }
    }
  }

  // abnormal termination
  /*virtual*/ void Operation::mark_terminated(void)
  {
    // no lock needed here - a running operation is exclusively owned by the runner
    assert(status.result == ProfilingMeasurements::OperationStatus::RUNNING);

    status.result = ProfilingMeasurements::OperationStatus::TERMINATED_EARLY;
    timeline.end_time = now();

    log_op.info("operation %p terminated: finish=" IDFMT "/%d",
		this, finish_event.id, finish_event.gen);

    // send out whatever profiling data we've gathered
    send_profiling_data();

    // trigger the finish event with poison
    trigger_finish_event(true /*poisoned!*/);
  }

  // successful termination
  /*virtual*/ void Operation::mark_completed(bool successful)
  {
    // no lock needed here - a running operation is exclusively owned by the runner
    assert(status.result == ProfilingMeasurements::OperationStatus::RUNNING);

    status.result = (successful ?
		       ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY :
		       ProfilingMeasurements::OperationStatus::COMPLETED_WITH_ERRORS);
    timeline.end_time = now();

    log_op.info("operation %p completed: finish=" IDFMT "/%d success=%d",
		this, finish_event.id, finish_event.gen, successful);

    // send out whatever profiling data we've gathered
    send_profiling_data();

    // trigger the finish event (poison if not completion was not successful)
    trigger_finish_event(!successful);
  }

  /*virtual*/ bool Operation::attempt_cancellation(int error_code)
  {
    if(!pmc.wants_measurement<Realm::ProfilingMeasurements::OperationStatus>()) {
      fprintf(stderr, "FATAL: Cancellation requested for a non-profiled task!\n");
      assert(0);
    }
    
    // if we catch the operation in either the WAITING or READY states, we can do the
    //  cancellation ourselves (don't forget to send profiling data!)
    // if the operation is running, we do nothing - a subclass might try something
    // if the operation is already completed or cancelled, we don't have to do anything
    {
      AutoHSLLock a(mutex);

      switch(status.result) {
      case ProfilingMeasurements::OperationStatus::WAITING:
      case ProfilingMeasurements::OperationStatus::READY:
	{
	  status.result = ProfilingMeasurements::OperationStatus::CANCELLED;
	  status.error_code = error_code;
	  break;  // we'll send profiling data below
	}

      case ProfilingMeasurements::OperationStatus::RUNNING:
	{
	  // let subclass try to handle this if it wants
	  return false;
	}

      case ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY:
      case ProfilingMeasurements::OperationStatus::COMPLETED_WITH_ERRORS:
      case ProfilingMeasurements::OperationStatus::TERMINATED_EARLY:
      case ProfilingMeasurements::OperationStatus::CANCELLED:
	{
	  // nothing for us, or subclass, to do
	  return false;
	}

	// no default here - want compiler warnings if we missed a state
      }
    }

    log_op.info("operation %p cancelled: finish=" IDFMT "/%d error=%d",
		this, finish_event.id, finish_event.gen, error_code);

    // we only get here if we did the cancellation - send profiling data now
    send_profiling_data();

    // trigger the finish event with poison
    trigger_finish_event(true /*poisoned!*/);

    // and tell caller we've taken care of everything
    return true;
  }

  void Operation::send_profiling_data(void)
  {
    if(prs) {
      // report profiling data
      pmc.add_measurement(status);
      pmc.add_measurement(timeline);
      pmc.send_responses(*prs);
    }
  }

  // NOTE: this must be the last thing done by an Operation - once the finish
  //  event is triggered, the Operation * may be deleted at any time
  void Operation::trigger_finish_event(bool poisoned)
  {
    GenEventImpl::local_trigger(finish_event, poisoned);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class OperationTable
  //

  // internal implementation is currently a single std::map
  // this will probably be a serial bottleneck - consider something more parallel

  struct OperationTableEntry {
    OperationTableEntry(Event _finish_event)
      : finish_event(_finish_event), local_op(0), 
	remote_node(-1), pending_cancellation(false)
    {}

    Event finish_event;
    Operation *local_op;
    int remote_node;
    bool pending_cancellation;
  };

  class OperationTable::Impl {
  public:
    GASNetHSL mutex;
    std::map<Event, OperationTableEntry> table;

    //void cancel_entry(Event finish_event, OperationTableEntry *entry, bool forward_ok);
  };

  class OperationTableCleaner : public LegionRuntime::LowLevel::EventWaiter {
  public:
    OperationTableCleaner(OperationTable::Impl *_impl, Event _event)
      : impl(_impl), event(_event)
    {}

    virtual ~OperationTableCleaner(void) {};

    virtual bool event_triggered(bool poisoned)
    {
      // we don't care about poisoned - the operation comes out of the table
      //  either way
      Operation *to_delete = 0;
      {
	AutoHSLLock a(impl->mutex);

	std::map<Event, OperationTableEntry>::iterator it = impl->table.find(event);
	// should always exist
	assert(it != impl->table.end());

	// delete local operation (if any) outside of lock
	to_delete = it->second.local_op;

	impl->table.erase(it);
      }

      log_optable.info("event " IDFMT "/%d cleaned",
		       event.id, event.gen);

      if(to_delete)
	to_delete->remove_reference(); // will be deleted if refcount is now 0

      return true;  // triggerer can delete us
    }

    virtual void print_info(FILE *f)
    {
      fprintf(f, "operation table cleaner: impl=%p event=" IDFMT "/%d\n",
	      impl, event.id, event.gen);
    }

  protected:
    OperationTable::Impl *impl;
    Event event;
  };

  struct OperationCancellationArgs {
    int sender;
    Event event;
  };

  void handle_operation_cancellation(OperationCancellationArgs args);

  enum { OPERATION_CANCEL_MSGID = 251 };

  typedef ActiveMessageShortNoReply<OPERATION_CANCEL_MSGID,
				    OperationCancellationArgs,
				    handle_operation_cancellation> OperationCancellationMessage;

  // global operation table
  /*extern*/ OperationTable operation_table;

  OperationTable::OperationTable(void)
  {
    impl = new Impl;
  }

  OperationTable::~OperationTable(void)
  {
    delete impl;
  }

  // Operations are 'owned' by the table - the table will free them once it
  //  gets the completion event for it
  void OperationTable::add_local_operation(Event finish_event, Operation *local_op)
  {
    OperationTableEntry new_entry(finish_event);
    new_entry.local_op = local_op;

    log_optable.info("event " IDFMT "/%d added: local_op=%p",
		     finish_event.id, finish_event.gen, local_op);

    // add a reference to the operation for our table entry
    local_op->add_reference();

    bool entry_added = false;
    bool cancel_immediately = false;
    {
      AutoHSLLock a(impl->mutex);

      // duplicates are possible if (and only if) a pending cancellation request exists
      std::map<Event, OperationTableEntry>::iterator it = impl->table.find(finish_event);
      if(it == impl->table.end()) {
	impl->table.insert(std::make_pair(finish_event, new_entry));
	entry_added = true;
      } else {
	// sanity check that it's just the pending cancellation
	assert(it->second.local_op == 0);
	assert(it->second.remote_node == -1);
	assert(it->second.pending_cancellation == true);

	// take a reference
	// we'll cancel outside the lock, so add another temp reference to the object
	// (one reference stays in here until the cleanup happens)
	local_op->add_reference();
	cancel_immediately = true;
	
	it->second.pending_cancellation = false;
	it->second.local_op = local_op;
      }
    }

    // register a cleanup event if we made a new entry
    if(entry_added)
      finish_event.impl()->add_waiter(finish_event.gen,
				      new OperationTableCleaner(impl, finish_event));

    // if there was a pending cancellation request, do that now
    if(cancel_immediately) {
      bool did_cancel = local_op->attempt_cancellation(Realm::Faults::ERROR_CANCELLED);
      log_optable.info("event " IDFMT "/%d - operation %p cancelled=%d",
		       finish_event.id, finish_event.gen, local_op, did_cancel);
      local_op->remove_reference();
    }
  }

  void OperationTable::add_remote_operation(Event finish_event, int remote_node)
  {
    OperationTableEntry new_entry(finish_event);
    new_entry.remote_node = remote_node;

    log_optable.info("event " IDFMT "/%d added: remote_node=%d",
		     finish_event.id, finish_event.gen, remote_node);

    {
      AutoHSLLock a(impl->mutex);

      // no duplicates please
      assert(impl->table.count(finish_event) == 0);
      impl->table.insert(std::make_pair(finish_event, new_entry));
    }

    // register a cleanup event
    finish_event.impl()->add_waiter(finish_event.gen,
				    new OperationTableCleaner(impl, finish_event));
  }

#if 0
  // handles the cancellation of an operation stored in an entry,
  //  dealing with the deferred free case
  void OperationTable::Impl::cancel_entry(Event finish_event,
					  OperationTableEntry *entry, bool forward_ok)
  {
    if(entry->local_op != 0) {
      entry->local_op->attempt_cancellation(Realm::Faults::ERROR_CANCELLED);
    }

    if(entry->remote_node != -1) {
      assert(forward_ok);

      log_optable.info("event " IDFMT"/%d cancellation forwarded to node %d",
		       finish_event.id, finish_event.gen, entry->remote_node);

      OperationCancellationArgs args;
      args.sender = gasnet_mynode();
      args.event = finish_event;
      OperationCancellationMessage::request(entry->remote_node, args);
    }

    // now decrement the refcount, and if we were the last, we clean up
    int count = __sync_add_and_fetch(&(entry->refcount), -1);
    if(count == 0) {
      if(entry->local_op)
	delete entry->local_op;

      {
	AutoHSLLock a(mutex);

	table.erase(finish_event);
      }

      log_optable.info("event " IDFMT "/%d delayed cleanup",
		       finish_event.id, finish_event.gen);
    }
  }
#endif

  void handle_operation_cancellation(OperationCancellationArgs args)
  {
    OperationTable::Impl *impl = operation_table.impl;

    // four cases:
    // 1) local operation is in table - cancel it
    // 2) remote operation is in table - die horribly (shouldn't happen)
    // 3) not in table and requestor was owner - assume messages arrived out of
    //      order and remember that we want to cancel operation when it arrives
    // 4) not in table and we are owner - assume it already finished and drop
    //      request
    Event finish_event = args.event;
    Operation *to_cancel = 0;
    {
      AutoHSLLock a(impl->mutex);

      std::map<Event, OperationTableEntry>::iterator it = impl->table.find(finish_event);
      if(it != impl->table.end()) {
	if(it->second.local_op != 0) {
	  // case 1
	  to_cancel = it->second.local_op;
	  to_cancel->add_reference();  // add a reference in case the event triggers simulataneously
	}

	if(it->second.remote_node != -1) {
	  // case 2
	  assert(0 && "should never receive a remote cancellation request for a non-local operation");
	}
      } else {
	int owner = LegionRuntime::LowLevel::ID(finish_event).node();

	if(args.sender == owner) {
	  // case 3
	  OperationTableEntry new_entry(finish_event);
	  new_entry.pending_cancellation = true;

	  impl->table.insert(std::make_pair(finish_event, new_entry));

	  log_optable.info("event " IDFMT"/%d cancellation from node %d - not in table yet?",
			   finish_event.id, finish_event.gen, args.sender);
	} else {
	  // case 4
	  assert(owner == gasnet_mynode());
	  log_optable.info("event " IDFMT"/%d cancellation from node %d ignored - not in table",
			   finish_event.id, finish_event.gen, args.sender);
	}
      }
    }

    if(to_cancel != 0) {
      bool did_cancel = to_cancel->attempt_cancellation(Realm::Faults::ERROR_CANCELLED);
      log_optable.info("event " IDFMT "/%d cancellation from node %d - operation %p cancelled=%d",
		       finish_event.id, finish_event.gen, args.sender, to_cancel, did_cancel);
      to_cancel->remove_reference();
    }
  }

  void OperationTable::request_cancellation(Event finish_event)
  {
    // look in the table and see if we have an entry - if so, keep a reference
    //  so that it doesn't get nuked once we let go of the lock
    bool found = false;
    Operation *to_cancel = 0;
    int forward_to = -1;
    {
      AutoHSLLock a(impl->mutex);

      std::map<Event, OperationTableEntry>::iterator it = impl->table.find(finish_event);
      if(it != impl->table.end()) {
	found = true;

	if(it->second.local_op) {
	  to_cancel = it->second.local_op;
	  to_cancel->add_reference();
	}

	if(it->second.remote_node != -1)
	  forward_to = it->second.remote_node;
      }
    }

    if(!found) {
      // if we get here, it's because the event wasn't in the table - that's
      //  either because we don't know about it (in which case we'll ask the node
      //  that created the event) or because it might have already triggered (which
      //  is what we'll assume if we're the node that created the event)

      int owner = LegionRuntime::LowLevel::ID(finish_event).node();

      if(owner == gasnet_mynode()) {
	// locally created event
	log_optable.info("event " IDFMT"/%d cancellation ignored - not in table",
			 finish_event.id, finish_event.gen);
	return;
      } else
	forward_to = owner;
    }

    if(to_cancel != 0) {
      bool did_cancel = to_cancel->attempt_cancellation(Realm::Faults::ERROR_CANCELLED);
      log_optable.info("event " IDFMT "/%d - operation %p cancelled=%d",
		       finish_event.id, finish_event.gen, to_cancel, did_cancel);
      to_cancel->remove_reference();
    }

    if(forward_to != -1) {
      log_optable.info("event " IDFMT"/%d cancellation forwarded to node %d",
		       finish_event.id, finish_event.gen, forward_to);

      OperationCancellationArgs args;
      args.sender = gasnet_mynode();
      args.event = finish_event;
      OperationCancellationMessage::request(forward_to, args);
    }
  }

  /*static*/ int OperationTable::register_handlers(gasnet_handlerentry_t *handlers)
  {
    int hcount = 0;
    hcount += OperationCancellationMessage::add_handler_entries(&handlers[hcount], "Operation Cancellation AM");
    return hcount;
  }

}; // namespace Realm

