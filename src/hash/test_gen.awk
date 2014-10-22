# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# awk -f test_gen.awk test_cases.txt
# generates test case table.
# edit next line to set particular reference implementation and name.
BEGIN { cmd = "echo -n `9 sha1sum`"; name = "Sha1Test" }
{
	printf("\t%s{ \"", name);
	printf("%s", $0) |cmd;
	close(cmd);
	printf("\", \"%s\" },\n", $0);
}
