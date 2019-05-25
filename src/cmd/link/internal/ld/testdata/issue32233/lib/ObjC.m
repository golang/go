// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#import <Foundation/Foundation.h>
#import <UserNotifications/UserNotifications.h>

BOOL function(void) {
  if (@available(macOS 10.14, *)) {
    UNUserNotificationCenter *center =
        [UNUserNotificationCenter currentNotificationCenter];
  }
  return NO;
}
