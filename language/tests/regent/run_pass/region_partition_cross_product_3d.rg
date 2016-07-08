-- Copyright 2016 Stanford University, NVIDIA Corporation
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Test: cross product between two partitions on structured index space (3D).

import "regent"

fspace access_info {
  -- This location is accessed through partition cp[dim1][dim2].
  { dim1_x, dim1_y, dim1_z } : int,
  { dim2_x, dim2_y, dim2_z } : int,
  -- #times location is accessed.
  count : int,
}

task main()
  var is = ispace(int3d, { x = 6, y = 6, z = 6 })

  -- Color spaces for partitions.
  var is_a = ispace(int3d, { x = 2, y = 3, z = 2 })
  var is_b = ispace(int3d, { x = 3, y = 2, z = 3 })

  var r = region(is, access_info)

  for i in is do
    r[i].count = 0
  end

  var pa = partition(equal, r, is_a)
  var pb = partition(equal, r, is_b)
  var cp = cross_product(pa, pb)
  
  -- Access each location through the cross product.
  for i1 in is_a do
    var p1 = cp[i1]
    for i2 in is_b do
      var p2 = p1[i2]
      for i in p2 do
        p2[i].dim1_x = i1.x
        p2[i].dim1_y = i1.y
        p2[i].dim1_z = i1.z
        p2[i].dim2_x = i2.x
        p2[i].dim2_y = i2.y
        p2[i].dim2_z = i2.z
        p2[i].count += 1
      end
    end
  end

  -- Verify access info.
  for i1 in is_a do
    var ra = pa[i1]
    for i in ra do
      regentlib.assert(ra[i].dim1_x == i1.x, "access index 1 x doesn't match")
      regentlib.assert(ra[i].dim1_y == i1.y, "access index 1 y doesn't match")
      regentlib.assert(ra[i].dim1_z == i1.z, "access index 1 z doesn't match")
    end
  end
  for i2 in is_b do
    var rb = pb[i2]
    for i in rb do
      regentlib.assert(rb[i].dim2_x == i2.x, "access index 2 x doesn't match")
      regentlib.assert(rb[i].dim2_y == i2.y, "access index 2 y doesn't match")
      regentlib.assert(rb[i].dim2_z == i2.z, "access index 2 z doesn't match")
    end
  end
  for i in is do
    regentlib.assert(r[i].count == 1, "access count doesn't match")
  end
end

regentlib.start(main)

