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

// defines the base Operation class used by all application-initiated operations as well
//  as the OperationTable, which is used to track down operations when cancellation requests are made

#ifndef REALM_OPERATION_H
#define REALM_OPERATION_H

#include "lowlevel.h"

#include "realm/profiling.h"
#include "activemsg.h"

namespace Realm {

  typedef LegionRuntime::LowLevel::Event Event;
  typedef LegionRuntime::LowLevel::GASNetHSL GASNetHSL;

  // TODO: move OperationStatus and OperationTimeline here?
  
  // an Operation has an Event that will be triggered when it completes, and
  //  includes the data structures necessary for doing basic profiling
  class Operation {
  protected:
    // can't construct an Operation directly
    // takes ownership of the ProfilingRequestSet - copy first if needed
    Operation(Event _finish_event, int _priority, ProfilingRequestSet *_prs);

    // can't destroy directly either - done when last reference is removed
    virtual ~Operation(void);
  public:
#ifdef DEBUG_REFCOUNTS
    void add_reference(void) { int x = __sync_add_and_fetch(&refcount, 1); printf("+ %p %d\n", this, x); }
    void remove_reference(void) { int x = __sync_add_and_fetch(&refcount, -1); printf("- %p %d\n", this, x); if(x == 0) delete this; }
#else
    void add_reference(void) { __sync_fetch_and_add(&refcount, 1); }
    void remove_reference(void) { if(__sync_add_and_fetch(&refcount, -1) == 0) delete this; }
#endif

    // these are called as the operation progresses through its life cycle

    // mark_ready() and mark_started() return false if they lose to a cancellation request 
    virtual bool mark_ready(void);
    virtual bool mark_started(void);
    virtual void mark_terminated(void);  // early termination
    virtual void mark_completed(bool successful);

    // returns true if its able to perform the cancellation (or if nothing can be done)
    // returns false if a subclass wants to try some other means to cancel an operation
    virtual bool attempt_cancellation(int error_code);

  protected:
    void send_profiling_data(void);

    // NOTE: this must be the last thing done by an Operation - once the finish
    //  event is triggered, the Operation * may be deleted at any time
    void trigger_finish_event(bool poisoned);
    
    GASNetHSL mutex;
    int refcount;

  public:
    Event finish_event;
    int priority;
    
    Realm::ProfilingRequestSet *prs;
    Realm::ProfilingMeasurementCollection pmc;
    Realm::ProfilingMeasurements::OperationStatus status;
    Realm::ProfilingMeasurements::OperationTimeline timeline;
  };

  class OperationTable {
  public:
    OperationTable(void);
    ~OperationTable(void);

    // Operations are 'owned' by the table - the table will free them once it
    //  gets the completion event for it
    void add_local_operation(Event finish_event, Operation *local_op);
    void add_remote_operation(Event finish_event, int remote_note);

    void request_cancellation(Event finish_event);
    
    static int register_handlers(gasnet_handlerentry_t *handlers);

    class Impl;
    Impl *impl;
  };

  // singleton object
  extern OperationTable operation_table;

}; // namespace Realm

#endif // REALM_OPERATION_H
