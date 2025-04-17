// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef struct Issue4339 Issue4339;

struct Issue4339 {
	char *name;
	void (*bar)(void);
};

extern Issue4339 exported4339;
void	handle4339(Issue4339*);
