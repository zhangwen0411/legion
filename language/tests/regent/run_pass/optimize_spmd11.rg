-- Copyright 2016 Stanford University
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

-- runs-with:
-- [
--   ["-ll:cpu", "4", "-fflow-spmd", "1"],
--   ["-ll:cpu", "2", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

-- This was supposed to be a reproducer for the MiniAero bug, but I can't seem to reproduce the issue.

-- This tests the SPMD optimization of the compiler with:
--   * aliased, disjoint partitions which open prior to loop
--   * read-only fields

local c = regentlib.c

struct t {
  a : int,
  b : int,
  c : int,
}

task inc_ba(r : region(t))
where reads(r.b), reads writes(r.a) do
  for x in r do
    x.a += x.b
  end
end

task avg_ac(r : region(t), q : region(t))
where reads(r.a, q.a), reads writes(r.c) do
  var s, t = 0, 0
  for y in q do
    s += y.a
    t += 1
  end
  var a = s / t
  for x in r do
    x.c += x.a - a
  end
end

task rot_cb(r : region(t), y : int, z : int)
where reads(r.c), reads writes(r.b) do
  for x in r do
    if x.c >= 0 then
      x.b = (x.c + y) % z
    else
      x.b = -((-x.c + y) % z)
    end
  end
end

task main()
  var r = region(ispace(ptr, 5), t)
  var x0 = new(ptr(t, r))
  var x1 = new(ptr(t, r))
  var x2 = new(ptr(t, r))
  var x3 = new(ptr(t, r))

  var cp = c.legion_coloring_create()
  c.legion_coloring_add_point(cp, 0, __raw(x0))
  c.legion_coloring_add_point(cp, 1, __raw(x1))
  c.legion_coloring_add_point(cp, 2, __raw(x2))
  c.legion_coloring_add_point(cp, 3, __raw(x3))
  var p = partition(disjoint, r, cp)
  c.legion_coloring_destroy(cp)

  var cq = c.legion_coloring_create()
  c.legion_coloring_add_point(cq, 0, __raw(x0))
  c.legion_coloring_add_point(cq, 0, __raw(x1))
  c.legion_coloring_add_point(cq, 1, __raw(x0))
  c.legion_coloring_add_point(cq, 1, __raw(x1))
  c.legion_coloring_add_point(cq, 1, __raw(x2))
  c.legion_coloring_add_point(cq, 2, __raw(x1))
  c.legion_coloring_add_point(cq, 2, __raw(x2))
  c.legion_coloring_add_point(cq, 2, __raw(x3))
  c.legion_coloring_add_point(cq, 3, __raw(x2))
  c.legion_coloring_add_point(cq, 3, __raw(x3))
  var q = partition(aliased, r, cq)
  c.legion_coloring_destroy(cq)

  var i = 0
  -- Note: This test hits a bug in the vectorizer if vectorization is allowed.
  __forbid(__vectorize)
  for x in r do
    x.a = 10000 + 10 * i
    x.b = 0
    x.c = 0
    i += 1
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  var pieces = 4

  for i = 0, pieces do
    inc_ba(p[i])
  end

  __demand(__spmd)
  for t = 0, 10 do
    for i = 0, pieces do
      avg_ac(p[i], q[i])
    end
  end

  for i = 0, pieces do
    rot_cb(p[i], 300, 137)
  end

  for i = 0, pieces do
    inc_ba(p[i])
  end

  for x in r do
    c.printf("x %d %d %d\n", x.a, x.b, x.c)
  end

  regentlib.assert(x0.a ==  9924, "test failed")
  regentlib.assert(x1.a == 10036, "test failed")
  regentlib.assert(x2.a == 10046, "test failed")
  regentlib.assert(x3.a == 10106, "test failed")
end
regentlib.start(main)
