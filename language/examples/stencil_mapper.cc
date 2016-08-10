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
                    std::map<Processor, Memory>* proc_sysmems);

      virtual void map_copy(const MapperContext      ctx,
                            const Copy&              copy,
                            const MapCopyInput&      input,
                                  MapCopyOutput&     output);
    private:
      void default_create_copy_instance(MapperContext ctx,
                     const Copy &copy, const RegionRequirement &req, 
                     const RegionRequirement &other_req, 
                     unsigned idx, std::vector<PhysicalInstance> &instances);

      std::vector<Processor>& procs_list;
      std::map<Processor, Memory>& proc_sysmems;
    };


    StencilMapper::StencilMapper(MapperRuntime *rt, Machine machine, Processor local,
                                 const char *mapper_name, std::vector<Processor>* _procs_list,
                                 std::map<Processor, Memory>* _proc_sysmems)
      : DefaultMapper(rt, machine, local, mapper_name),
        procs_list(*_procs_list), proc_sysmems(*_proc_sysmems)
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
      // std::cout << cr << ", " << index << std::endl;

      Processor target_proc = procs_list[index % procs_list.size()];
      Memory target_memory = proc_sysmems[target_proc];

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
  };
};

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  using namespace Legion::Mapping;

  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();

  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        (*proc_sysmems)[affinity.p] = affinity.m;
      }
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
  }

  // std::cout << "procs_list.size() is " << procs_list->size() << std::endl;

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); ++it)
  {
    StencilMapper* mapper = new StencilMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "stencil_mapper",
                                              procs_list, proc_sysmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}
