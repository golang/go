// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#import <Foundation/Foundation.h>
#import <AppKit/NSAppearance.h>

BOOL function(void) {
#if defined(MAC_OS_X_VERSION_MIN_REQUIRED) && (MAC_OS_X_VERSION_MIN_REQUIRED > 101300)
  NSAppearance *darkAppearance;
  if (@available(macOS 10.14, *)) {
    darkAppearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
  }
#endif
  return NO;
}
