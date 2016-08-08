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

-- Inspired by https://github.com/ParRes/Kernels/tree/master/LEGION/Stencil

import "regent"

local common = require("stencil_common")

local DTYPE = double
local RADIUS = 2

local c = regentlib.c

local min = regentlib.fmin
local max = regentlib.fmax

fspace point {
  input : DTYPE,
  output : DTYPE,
}

terra to_rect(lo : int2d, hi : int2d) : c.legion_rect_2d_t
  return c.legion_rect_2d_t {
    lo = lo:to_point(),
    hi = hi:to_point(),
  }
end

task make_private_partition(points : region(ispace(int2d), point),
                            tiles : ispace(int2d),
                            n : int2d, nt : int2d, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var npoints = n + nt*{ 2*radius, 2*radius }
  for i in tiles do
    var lo = int2d { x = i.x*npoints.x/nt.x, y = i.y*npoints.y/nt.y }
    var hi = int2d { x = (i.x+1)*npoints.x/nt.x - 1, y = (i.y+1)*npoints.y/nt.y - 1 }
    var rect = to_rect(lo, hi)
    c.legion_domain_point_coloring_color_domain(
      coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_interior_partition(points : region(ispace(int2d), point),
                             tiles : ispace(int2d),
                             n : int2d, nt : int2d, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var npoints = n + nt*{ 2*radius, 2*radius }
  for i in tiles do
    var lo = int2d { x = i.x*npoints.x/nt.x + radius, y = i.y*npoints.y/nt.y + radius }
    var hi = int2d { x = (i.x+1)*npoints.x/nt.x - 1 - radius, y = (i.y+1)*npoints.y/nt.y - 1 - radius }
    var rect = to_rect(lo, hi)
    c.legion_domain_point_coloring_color_domain(
      coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_exterior_partition(points : region(ispace(int2d), point),
                             tiles : ispace(int2d),
                             n : int2d, nt : int2d, radius : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var npoints = n + nt*{ 2*radius, 2*radius }
  for i in tiles do
    var loffx, loffy = radius, radius
    if i.x == 0 then loffx = 0 end
    if i.y == 0 then loffy = 0 end

    var hoffx, hoffy = radius, radius
    if i.x == nt.x - 1 then hoffx = 0 end
    if i.y == nt.y - 1 then hoffy = 0 end

    var lo = int2d { x = i.x*npoints.x/nt.x + loffx, y = i.y*npoints.y/nt.y + loffy }
    var hi = int2d { x = (i.x+1)*npoints.x/nt.x - 1 - hoffx, y = (i.y+1)*npoints.y/nt.y - 1 - hoffy }
    var rect = to_rect(lo, hi)
    c.legion_domain_point_coloring_color_domain(
      coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

terra clamp(val : int64, lo : int64, hi : int64)
  return min(max(val, lo), hi)
end


function make_ghost_x_partition(is_complete)
  local task ghost_x_partition(points : region(ispace(int2d), point),
                               tiles : ispace(int2d),
                               n : int2d, nt : int2d, radius : int64,
                               dir : int64)
    var coloring = c.legion_domain_point_coloring_create()
    for i in tiles do
      var lo = int2d { x = clamp((i.x+dir)*radius, 0, nt.x*radius), y = i.y*n.y/nt.y }
      var hi = int2d { x = clamp((i.x+1+dir)*radius - 1, -1, nt.x*radius - 1), y = (i.y+1)*n.y/nt.y - 1 }
      var rect = to_rect(lo, hi)
      c.legion_domain_point_coloring_color_domain(
        coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
    end
    var p = [(
        function()
          if is_complete then
            return rexpr partition(disjoint, points, coloring, tiles) end
          else
            -- Hack: Since the compiler does not track completeness as
            -- a static property, mark incomplete partitions as aliased.
            return rexpr partition(aliased, points, coloring, tiles) end
          end
        end)()]
    c.legion_domain_point_coloring_destroy(coloring)
    return p
  end
  return ghost_x_partition
end

function make_ghost_y_partition(is_complete)
  local task ghost_y_partition(points : region(ispace(int2d), point),
                               tiles : ispace(int2d),
                               n : int2d, nt : int2d, radius : int64,
                               dir : int64)
    var coloring = c.legion_domain_point_coloring_create()
    for i in tiles do
      var lo = int2d { x = i.x*n.x/nt.x, y = clamp((i.y+dir)*radius, 0, nt.y*radius) }
      var hi = int2d { x = (i.x+1)*n.x/nt.x - 1, y = clamp((i.y+1+dir)*radius - 1, -1, nt.y*radius - 1) }
      var rect = to_rect(lo, hi)
      c.legion_domain_point_coloring_color_domain(
        coloring, i:to_domain_point(), c.legion_domain_from_rect_2d(rect))
    end
    var p = [(
        function()
          if is_complete then
            return rexpr partition(disjoint, points, coloring, tiles) end
          else
            -- Hack: Since the compiler does not track completeness as
            -- a static property, mark incomplete partitions as aliased.
            return rexpr partition(aliased, points, coloring, tiles) end
          end
        end)()]
    c.legion_domain_point_coloring_destroy(coloring)
    return p
  end
  return ghost_y_partition
end

task stencil(private : region(ispace(int2d), point),
             interior : region(ispace(int2d), point),
             xm : region(ispace(int2d), point),
             xp : region(ispace(int2d), point),
             ym : region(ispace(int2d), point),
             yp : region(ispace(int2d), point),
             print_ts : bool)
where
  reads writes(private.{input, output}),
  reads(xm.input, xp.input, ym.input, yp.input)
do
  if print_ts then c.printf("t: %ld\n", c.legion_get_current_time_in_micros()) end
end

task increment(private : region(ispace(int2d), point),
               exterior : region(ispace(int2d), point),
               xm : region(ispace(int2d), point),
               xp : region(ispace(int2d), point),
               ym : region(ispace(int2d), point),
               yp : region(ispace(int2d), point),
               print_ts : bool)
where reads writes(private.input, xm.input, xp.input, ym.input, yp.input) do
  if print_ts then c.printf("t: %ld\n", c.legion_get_current_time_in_micros()) end
end

task check(private : region(ispace(int2d), point),
           interior : region(ispace(int2d), point),
           tsteps : int64, init : int64)
where reads(private.{input, output}) do
  c.printf("done\n")
end

task main()
  var conf = common.read_config()

  var nbloated = int2d { conf.nx, conf.ny } -- Grid size along each dimension, including border.
  var nt = int2d { conf.ntx, conf.nty } -- Number of tiles to make in each dimension.
  var init : int64 = conf.init

  var radius : int64 = RADIUS
  var n = nbloated - { 2*radius, 2*radius } -- Grid size, minus the border.
  regentlib.assert(n >= nt, "grid too small")

  var grid = ispace(int2d, n + nt*{ 2*radius, 2*radius })
  var tiles = ispace(int2d, nt)

  var points = region(grid, point)
  var private = make_private_partition(points, tiles, n, nt, radius)
  var interior = make_interior_partition(points, tiles, n, nt, radius)
  var exterior = make_exterior_partition(points, tiles, n, nt, radius)

  var xm = region(ispace(int2d, { x = nt.x*radius, y = n.y }), point)
  var xp = region(ispace(int2d, { x = nt.x*radius, y = n.y }), point)
  var ym = region(ispace(int2d, { x = n.x, y = nt.y*radius }), point)
  var yp = region(ispace(int2d, { x = n.x, y = nt.y*radius }), point)
  var pxm_in = [make_ghost_x_partition(false)](xm, tiles, n, nt, radius, -1)
  var pxp_in = [make_ghost_x_partition(false)](xp, tiles, n, nt, radius,  1)
  var pym_in = [make_ghost_y_partition(false)](ym, tiles, n, nt, radius, -1)
  var pyp_in = [make_ghost_y_partition(false)](yp, tiles, n, nt, radius,  1)
  var pxm_out = [make_ghost_x_partition(true)](xm, tiles, n, nt, radius, 0)
  var pxp_out = [make_ghost_x_partition(true)](xp, tiles, n, nt, radius, 0)
  var pym_out = [make_ghost_y_partition(true)](ym, tiles, n, nt, radius, 0)
  var pyp_out = [make_ghost_y_partition(true)](yp, tiles, n, nt, radius, 0)

  fill(points.{input, output}, init)
  fill(xm.{input, output}, init)
  fill(xp.{input, output}, init)
  fill(ym.{input, output}, init)
  fill(yp.{input, output}, init)

  var tsteps : int64 = conf.tsteps
  var tprune : int64 = conf.tprune
  regentlib.assert(tsteps > 2*tprune, "too few time steps")

  __demand(__spmd)
  do
    for t = 0, tsteps do
      -- __demand(__parallel)
      for i in tiles do
        stencil(private[i], interior[i], pxm_in[i], pxp_in[i], pym_in[i], pyp_in[i], t == tprune)
      end
      -- __demand(__parallel)
      for i in tiles do
        increment(private[i], exterior[i], pxm_out[i], pxp_out[i], pym_out[i], pyp_out[i], t == tsteps - tprune - 1)
      end
    end

    --[[
    for i in tiles do
      check(private[i], interior[i], tsteps, init)
    end
    ]]
  end
end
regentlib.start(main)

