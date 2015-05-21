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

// helper defines/data structures for fault reporting/handling in Realm

#ifndef REALM_FAULTS_H
#define REALM_FAULTS_H

namespace Realm {

  namespace Faults {

    // faults are reported with an integer error code - we define a bunch
    // here, and leave room for application-defined codes as well
    enum {
      ERROR_POISONED_EVENT = 1,     // querying a poisoned event without handling poison
      ERROR_POISONED_PRECONDITION,  // precondition to an operation was poisoned
      ERROR_CANCELLED,              // cancelled by request from application

      // application can use its own error codes too, but start
      //  here so we don't get any conflicts
      ERROR_APPLICATION_DEFINED = 1000,
    };

  }; // namespace Faults

}; // namespace Realm

#endif // REALM_FAULTS_H
