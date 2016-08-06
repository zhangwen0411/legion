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

import "regent"

struct s { a : int }

local r = regentlib.newsymbol(region(s), "r")
local reads_r_a = regentlib.privilege(regentlib.reads, r, "a")
local writes_r_a = regentlib.privilege(regentlib.writes, r, "a")
local reads_writes_r_a = terralib.newlist({reads_r_a, writes_r_a})

task f([r])
where [reads_writes_r_a] do
  for i in r do
    i.a += 1
  end
end

task main()
  var r = region(ispace(ptr, 5), s)
  new(ptr(s, r), 3)
  fill(r.a, 10)
  f(r)

  var t = 0
  for i in r do
    t += i.a
  end
  regentlib.assert(t == 33, "test failed")
end
regentlib.start(main)
