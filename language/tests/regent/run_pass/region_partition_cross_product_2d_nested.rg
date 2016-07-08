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

-- Test: cross product between multiple partitions on structured index space
-- (2D).

import "regent"

fspace access_info {
  -- This location is accessed through partition cp[dim1][dim2][dim3].
  { dim1_x, dim1_y } : int,
  { dim2_x, dim2_y } : int,
  { dim3_x, dim3_y } : int,
  -- #times location is accessed.
  count : int,
}

task main()
  var is = ispace(int2d, { x = 10, y = 10 })

  -- Color spaces for partitions.
  var is_a = ispace(int2d, { x = 5, y = 1 })
  var is_b = ispace(int2d, { x = 1, y = 5 })
  var is_c = ispace(int2d, { x = 2, y = 2 })

  var r = region(is, access_info)

  for i in is do
    r[i].count = 0
  end

  var pa = partition(equal, r, is_a)
  var pb = partition(equal, r, is_b)
  var pc = partition(equal, r, is_c)
  var cp = cross_product(pa, pb, pc)
  
  -- Access each location through the cross product.
  for i1 in is_a do
    var p1 = cp[i1]
    for i2 in is_b do
      var p2 = p1[i2]
      for i3 in is_c do
        var p3 = p2[i3]
        for i in p3 do
          p3[i].dim1_x = i1.x
          p3[i].dim1_y = i1.y
          p3[i].dim2_x = i2.x
          p3[i].dim2_y = i2.y
          p3[i].dim3_x = i3.y
          p3[i].dim3_y = i3.y
          p3[i].count += 1
        end
      end
    end
  end

  -- Verify access info.
  for i1 in is_a do
    var ra = pa[i1]
    for i in ra do
      regentlib.assert(ra[i].dim1_x == i1.x, "access index 1 x doesn't match")
      regentlib.assert(ra[i].dim1_y == i1.y, "access index 1 y doesn't match")
    end
  end
  for i2 in is_b do
    var rb = pb[i2]
    for i in rb do
      regentlib.assert(rb[i].dim2_x == i2.x, "access index 2 x doesn't match")
      regentlib.assert(rb[i].dim2_y == i2.y, "access index 2 y doesn't match")
    end
  end
  for i in is do
    regentlib.assert(r[i].count == 1, "access count doesn't match")
  end
end

regentlib.start(main)
