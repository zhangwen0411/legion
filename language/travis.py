#!/usr/bin/env python

# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os, platform, subprocess

def install_terra():
    platform_name = platform.system() if platform.system() != 'Darwin' else 'OSX'
    base_url = 'https://github.com/zdevito/terra/releases/download/release-2016-03-25'
    terra_tarball = 'terra-%s-x86_64-332a506.zip' % platform_name
    terra_url = '%s/%s' % (base_url, terra_tarball)
    terra_dir = os.path.abspath(os.path.splitext(terra_tarball)[0])

    shasums = {'Linux': '6a1c29a061c502aaf69d39c6a0f54702dea3ff60', 'Darwin': '8357b07b0bed33eac1355b55e135196119ec83ba'}

    subprocess.check_call(['curl', '-L', '-O', terra_url])
    shasum = subprocess.Popen(['shasum', '-c'], stdin=subprocess.PIPE)
    shasum.communicate(
        '%s  %s' % (shasums[platform.system()], terra_tarball))
    assert shasum.wait() == 0
    subprocess.check_call(['unzip', terra_tarball])

    return terra_dir

def test(root_dir, terra_dir, debug, env):
    subprocess.check_call(
        ['time', './install.py', '-j', '2', '--with-terra', terra_dir] +
        (['--debug'] if debug else []),
        env = env,
        cwd = root_dir)
    subprocess.check_call(
        ['time', './test.py', '-q', '-j', '2'] + (['--debug'] if debug else []),
        env = env,
        cwd = root_dir)

if __name__ == '__main__':
    root_dir = os.path.realpath(os.path.dirname(__file__))
    terra_dir = install_terra()

    # reduce output spewage by default
    env = dict(os.environ.iteritems())
    if 'MAKEFLAGS' not in env:
        env['MAKEFLAGS'] = 's'

    test(root_dir, terra_dir, env['DEBUG'], env)
