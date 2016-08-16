/* Copyright 2016 Stanford University
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

#include "stencil_mapper.h"
#include <iostream>
#include <vector>
#include <set>
#include <map>

#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;


namespace Legion {
  namespace Mapping {

    static LegionRuntime::Logger::Category log_mapper("stencil");

    class StencilMapper : public DefaultMapper
    {
    public:
      StencilMapper(MapperRuntime *rt, Machine machine, Processor local,
                    const char *mapper_name, std::vector<Processor>* procs_list,
                    std::vector<Memory>* sysmems_list,
                    std::map<Processor, Memory>* proc_sysmems,
                    std::map<Memory, std::vector<Processor> >* sysmem_local_procs,
                    std::map<Processor, Memory>* proc_target_mems);

      virtual void map_copy(const MapperContext      ctx,
                            const Copy&              copy,
                            const MapCopyInput&      input,
                                  MapCopyOutput&     output);
      virtual void map_must_epoch(const MapperContext           ctx,
                                  const MapMustEpochInput&      input,
                                  MapMustEpochOutput&     output);
      virtual Memory default_policy_select_target_memory(MapperContext ctx, 
                                        Processor target_proc);

    private:
      void default_create_copy_instance(MapperContext ctx,
                     const Copy &copy, const RegionRequirement &req, 
                     const RegionRequirement &other_req, 
                     unsigned idx, std::vector<PhysicalInstance> &instances);

      std::vector<Processor>& procs_list;
      std::vector<Memory>& sysmems_list;
      std::map<Processor, Memory>& proc_sysmems;
      std::map<Memory, std::vector<Processor> >& sysmem_local_procs;
      std::map<Processor, Memory>& proc_target_mems;
    };


    StencilMapper::StencilMapper(MapperRuntime *rt, Machine machine, Processor local,
                                 const char *mapper_name, std::vector<Processor>* _procs_list,
                                 std::vector<Memory>* _sysmems_list,
                                 std::map<Processor, Memory>* _proc_sysmems,
                                 std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
                                 std::map<Processor, Memory>* _proc_target_mems)
      : DefaultMapper(rt, machine, local, mapper_name),
        procs_list(*_procs_list), sysmems_list(*_sysmems_list),
        proc_sysmems(*_proc_sysmems), sysmem_local_procs(*_sysmem_local_procs),
        proc_target_mems(*_proc_target_mems)
    {
    }

    void StencilMapper::map_copy(const MapperContext      ctx,
                                 const Copy&              copy,
                                 const MapCopyInput&      input,
                                       MapCopyOutput&     output)
    {
      if (strcmp(copy.parent_task->get_task_name(), "main") != 0)
        return DefaultMapper::map_copy(ctx, copy, input, output);

      // For the sources always use an existing instances and virtual
      // instances for the rest, for the destinations, hope they are
      // restricted, otherwise we really don't know what to do
      bool has_unrestricted = false;
      for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      {
        output.src_instances[idx] = input.src_instances[idx];
        if (!output.src_instances[idx].empty())
          runtime->acquire_and_filter_instances(ctx,
                                              output.src_instances[idx]);
        // Check to see if we are doing a reduce-across in which case we
        // need to actually create a real physical instance
        if (copy.dst_requirements[idx].privilege == REDUCE)
        {
          // If the source is restricted, we know we are good
          if (!copy.src_requirements[idx].is_restricted())
            default_create_copy_instance(ctx, copy, 
                copy.src_requirements[idx], copy.dst_requirements[idx], idx, output.src_instances[idx]);
        }
        else // Stick this on for good measure, at worst it will be ignored
          output.src_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
        output.dst_instances[idx] = input.dst_instances[idx];
        if (!output.dst_instances[idx].empty())
          runtime->acquire_and_filter_instances(ctx,
                                  output.dst_instances[idx]);
        if (!copy.dst_requirements[idx].is_restricted())
          has_unrestricted = true;
      }
      // If the destinations were all restricted we know we got everything
      if (has_unrestricted)
      {
        for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
        {
          output.dst_instances[idx] = input.dst_instances[idx];
          if (!copy.dst_requirements[idx].is_restricted())
            default_create_copy_instance(ctx, copy, 
                copy.dst_requirements[idx], copy.src_requirements[idx], idx, output.dst_instances[idx]);
        }
      }
    }

    void StencilMapper::default_create_copy_instance(MapperContext ctx,
                         const Copy &copy, const RegionRequirement &req, 
                         const RegionRequirement &other_req, 
                         unsigned idx, std::vector<PhysicalInstance> &instances) 
    //--------------------------------------------------------------------------
    {
      // See if we have all the fields covered
      std::set<FieldID> missing_fields = req.privilege_fields;
      for (std::vector<PhysicalInstance>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        it->remove_space_fields(missing_fields);
        if (missing_fields.empty())
          break;
      }
      if (missing_fields.empty())
        return;

      bool has_parent1 = runtime->has_parent_logical_partition(ctx, req.region);
      bool has_parent2 = runtime->has_parent_logical_partition(ctx, other_req.region);
      assert(has_parent1 || has_parent2);
      LogicalPartition lp = runtime->get_parent_logical_partition(ctx, 
          has_parent1 ? req.region : other_req.region);
      IndexPartition ip = lp.get_index_partition();
      Domain color_space = runtime->get_index_partition_color_space(ctx, ip);
      DomainPoint color = runtime->get_logical_region_color_point(ctx, req.region);
      assert(color_space.contains(color));

      Rect<2> csr = color_space.get_rect<2>();
      Point<2> cr = color.get_point<2>();
      int index = cr[1] + cr[0] * csr.dim_size(1);

      /*
      Processor target_proc = procs_list[index % procs_list.size()];
      Memory target_memory = proc_sysmems[target_proc];
      */
      const unsigned long shardsize = 8;
      Memory target_memory = sysmems_list[(index / shardsize) % sysmems_list.size()];
      // std::cout << cr << ", " << index << ", " << target_memory << std::endl;

      bool force_new_instances = false;
      LayoutConstraintID our_layout_id = 
       default_policy_select_layout_constraints(ctx, target_memory, 
                                                req, COPY_MAPPING,
                                                true/*needs check*/, 
                                                force_new_instances);
      LayoutConstraintSet creation_constraints = 
                  runtime->find_layout_constraints(ctx, our_layout_id);
      creation_constraints.add_constraint(
          FieldConstraint(missing_fields,
                          false/*contig*/, false/*inorder*/));
      instances.resize(instances.size() + 1);
      if (!default_make_instance(ctx, target_memory, 
            creation_constraints, instances.back(), 
            COPY_MAPPING, force_new_instances, true/*meets*/, req))
      {
        // If we failed to make it that is bad
        log_mapper.error("Oops!");
        assert(false);
      }
    }


    void StencilMapper::map_must_epoch(const MapperContext           ctx,
                                       const MapMustEpochInput&      input,
                                       MapMustEpochOutput&     output)
    {
      std::vector<Processor > all_procs;
      for (std::map<Memory, std::vector<Processor> >::iterator it = sysmem_local_procs.begin();
           it != sysmem_local_procs.end(); ++it)
      {
        if (!it->second.empty()) all_procs.push_back(it->second[0]);
      }
      unsigned num_sysmems = all_procs.size();

      log_mapper.spew("Default map_must_epoch in %s", get_mapper_name());
      // Figure out how to assign tasks to CPUs first. We know we can't
      // do must epochs for anthing but CPUs at the moment.
      std::map<const Task*,Processor> proc_map;
      if (total_nodes > 1)
      {
        if (input.tasks.size() > num_sysmems)
        {
          log_mapper.error("Default mapper error. Not enough nodes for must "
                           "epoch launch of task %s with %ld tasks", 
                           input.tasks[0]->get_task_name(),
                           input.tasks.size());
          assert(false);
        }

        for (unsigned idx = 0; idx < input.tasks.size(); idx++)
        {
          output.task_processors[idx] = all_procs[idx];
          proc_map[input.tasks[idx]] = all_procs[idx];

          // std::cout << proc << std::endl;
        }
      }
      else
      {
        if (input.tasks.size() > local_cpus.size())
        {
          log_mapper.error("Default mapper error. Not enough CPUs for must "
                           "epoch launch of task %s with %ld tasks", 
                           input.tasks[0]->get_task_name(),
                           input.tasks.size());
          assert(false);
        }
        for (unsigned idx = 0; idx < input.tasks.size(); idx++)
        {
          output.task_processors[idx] = local_cpus[idx];
          proc_map[input.tasks[idx]] = local_cpus[idx];
        }
      }
      // Now let's map the constraints, find one requirement to use for
      // mapping each of the constraints, but get the set of fields we
      // care about and the set of logical regions for all the requirements
      // printf("num iters: %lu\n", input.constraints.size());
      // printf("start: %llu\n", Realm::Clock::current_time_in_microseconds());
      for (unsigned cid = 0; cid < input.constraints.size(); cid++)
      {
        const MappingConstraint &constraint = input.constraints[cid];
        std::vector<PhysicalInstance> &constraint_mapping = 
                                              output.constraint_mappings[cid];
        // Figure out which task and region requirement to use as the 
        // basis for doing the mapping
        Task *base_task = NULL;
        unsigned base_index = 0;
        Processor base_proc = Processor::NO_PROC;
        std::set<LogicalRegion> needed_regions;
        std::set<FieldID> needed_fields;
        for (unsigned idx = 0; idx < constraint.constrained_tasks.size(); idx++)
        {
          Task *task = constraint.constrained_tasks[idx];
          unsigned req_idx = constraint.requirement_indexes[idx];
          if ((base_task == NULL) && (!task->regions[req_idx].is_no_access()))
          {
            base_task = task;
            base_index = req_idx;
            base_proc = proc_map[task];
          }
          needed_regions.insert(task->regions[req_idx].region);
          needed_fields.insert(task->regions[req_idx].privilege_fields.begin(),
                               task->regions[req_idx].privilege_fields.end());
        }
        // If there wasn't a region requirement that wasn't no access just 
        // pick the first one since this case doesn't make much sense anyway
        if (base_task == NULL)
        {
          base_task = constraint.constrained_tasks[0];
          base_index = constraint.requirement_indexes[0];
          base_proc = proc_map[base_task];
        }
        Memory target_memory = default_policy_select_target_memory(ctx, 
                                                                   base_proc);
        VariantInfo info = default_find_preferred_variant(*base_task, ctx, 
               true/*needs tight bound*/, true/*cache*/, Processor::LOC_PROC);
        const TaskLayoutConstraintSet &layout_constraints = 
          runtime->find_task_layout_constraints(ctx, base_task->task_id, 
                                                info.variant);
        if (needed_regions.size() == 1)
        {
          // If there was just one region we can use the base region requirement
          if (!default_create_custom_instances(ctx, base_proc, target_memory,
                base_task->regions[base_index], base_index, needed_fields,
                layout_constraints, true/*needs check*/, constraint_mapping))
          {
            log_mapper.error("Default mapper error. Unable to make instance(s) "
                             "in memory " IDFMT " for index %d of constrained "
                             "task %s (ID %lld) in must epoch launch.",
                             target_memory.id, base_index,
                             base_task->get_task_name(), 
                             base_task->get_unique_id());
            assert(false);
          }
        }
        else
        {
          // Otherwise we need to find a common region that will satisfy all
          // the needed regions
          RegionRequirement copy_req = base_task->regions[base_index];
          copy_req.region = default_find_common_ancestor(ctx, needed_regions);
          if (!default_create_custom_instances(ctx, base_proc, target_memory,
                copy_req, base_index, needed_fields, layout_constraints,
                true/*needs check*/, constraint_mapping))
          {
            log_mapper.error("Default mapper error. Unable to make instance(s) "
                             "in memory " IDFMT " for index %d of constrained "
                             "task %s (ID %lld) in must epoch launch.",
                             target_memory.id, base_index,
                             base_task->get_task_name(), 
                             base_task->get_unique_id());
            assert(false);
          }
        }
      }
      // printf("finish: %llu\n", Realm::Clock::current_time_in_microseconds());
    }

    Memory StencilMapper::default_policy_select_target_memory(MapperContext ctx, 
                                                              Processor target_proc)
    {
      Memory target_mem = proc_target_mems[target_proc];
      assert(target_mem.exists());
      return target_mem;
    }
  };
};

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  using namespace Legion::Mapping;

  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_regmems = new std::map<Processor, Memory>();

  std::map<Processor, Memory>* proc_target_mems = new std::map<Processor, Memory>();
  std::map<Processor, unsigned> proc_best_bandwidth;

  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
        if (proc_regmems->find(affinity.p) == proc_regmems->end())
          (*proc_regmems)[affinity.p] = affinity.m;
      }
      else if (affinity.m.kind() == Memory::REGDMA_MEM)
        (*proc_regmems)[affinity.p] = affinity.m;
    }

    if (affinity.bandwidth >= proc_best_bandwidth[affinity.p]) {
      proc_best_bandwidth[affinity.p] = affinity.bandwidth;
      (*proc_target_mems)[affinity.p] = affinity.m;
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
    (*sysmem_local_procs)[it->second].push_back(it->first);
  }

  for (std::map<Memory, std::vector<Processor> >::iterator it =
        sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
    sysmems_list->push_back(it->first);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); ++it)
  {
    StencilMapper* mapper = new StencilMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "stencil_mapper",
                                              procs_list, sysmems_list, proc_sysmems,
                                              sysmem_local_procs, proc_target_mems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}
