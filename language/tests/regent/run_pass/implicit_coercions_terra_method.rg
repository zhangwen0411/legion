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

struct s
{
  x : int,
  y : int,
}
terra s:f(x : &int)
  x[0] = x[0] + self.x
  x[1] = x[1] + self.y
end

task main()
  var a : s = {1, 2}
  var y : int[2]
  y[0], y[1] = 10, 20
  a:f(y) -- The implicit cast from int[2] to &int must preserve l-val-ness.
  regentlib.assert(y[0] == 11, "test failed")
  regentlib.assert(y[1] == 22, "test failed")
end
regentlib.start(main)
