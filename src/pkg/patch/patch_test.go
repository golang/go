// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package patch

// TODO(rsc): test Apply

import "testing"

type Test struct {
	in   string
	out  string
	diff string
}

func TestFileApply(t *testing.T) {
	for i, test := range tests {
		set, err := Parse([]byte(test.diff))
		if err != nil {
			t.Errorf("#%d: Parse: %s", i, err)
			continue
		}
		if len(set.File) != 1 {
			t.Errorf("#%d: Parse returned %d patches, want 1", i, len(set.File))
			continue
		}
		new, err := set.File[0].Apply([]byte(test.in))
		if err != nil {
			t.Errorf("#%d: Apply: %s", i, err)
			continue
		}
		if s := string(new); s != test.out {
			t.Errorf("#%d:\n--- have\n%s--- want\n%s", i, s, test.out)
		}
	}
}

var tests = []Test{
	{
		"hello, world\n",
		"goodbye, world\n",
		"Index: a\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1 +1 @@\n" +
			"-hello, world\n" +
			"+goodbye, world\n",
	},
	{
		"hello, world\n",
		"goodbye, world\n",
		"Index: a\n" +
			"index cb34d9b1743b7c410fa750be8a58eb355987110b..0a01764bc1b2fd29da317f72208f462ad342400f\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1 +1 @@\n" +
			"-hello, world\n" +
			"+goodbye, world\n",
	},
	{
		"hello, world\n",
		"goodbye, world\n",
		"diff a/a b/b\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1,1 +1,1 @@\n" +
			"-hello, world\n" +
			"+goodbye, world\n",
	},
	{
		"hello, world",
		"goodbye, world\n",
		"diff --git a/a b/b\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1 +1 @@\n" +
			"-hello, world\n" +
			"\\ No newline at end of file\n" +
			"+goodbye, world\n",
	},
	{
		"hello, world\n",
		"goodbye, world",
		"Index: a\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1 +1 @@\n" +
			"-hello, world\n" +
			"+goodbye, world\n" +
			"\\ No newline at end of file\n",
	},
	{
		"hello, world",
		"goodbye, world",
		"Index: a\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1 +1 @@\n" +
			"-hello, world\n" +
			"\\ No newline at end of file\n" +
			"+goodbye, world\n" +
			"\\ No newline at end of file\n",
	},
	{
		"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\n",
		"a\nB\nC\nD\ne\nf\ng\nj\nk\nl\nm\nN\n",
		"Index: a\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1,14 +1,12 @@\n" +
			" a\n" +
			"-b\n" +
			"-c\n" +
			"-d\n" +
			"+B\n" +
			"+C\n" +
			"+D\n" +
			" e\n" +
			" f\n" +
			" g\n" +
			"-h\n" +
			"-i\n" +
			" j\n" +
			" k\n" +
			" l\n" +
			" m\n" +
			"-n\n" +
			"+N\n",
	},
	{
		"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n",
		"a\nb\nc\ng\nh\ni\nj\nk\nl\nm\nN\nO\np\nq\nr\ns\nt\nu\nv\nw\nd\ne\nf\nx\n",
		"Index: a\n" +
			"--- a/a\n" +
			"+++ b/b\n" +
			"@@ -1,9 +1,6 @@\n" +
			" a\n" +
			" b\n" +
			" c\n" +
			"-d\n" +
			"-e\n" +
			"-f\n" +
			" g\n" +
			" h\n" +
			" i\n" +
			"@@ -11,8 +8,8 @@ j\n" +
			" k\n" +
			" l\n" +
			" m\n" +
			"-n\n" +
			"-o\n" +
			"+N\n" +
			"+O\n" +
			" p\n" +
			" q\n" +
			" r\n" +
			"\n" +
			"@@ -21,6 +18,7 @@ t\n" +
			" u\n" +
			" v\n" +
			" w\n" +
			"+d\n" +
			"+e\n" +
			"+f\n" +
			" x\n" +
			"-y\n" +
			"-z\n",
	},
	{
		"a\nb\nc\ng\nh\ni\nj\nk\nl\nm\nN\nO\np\nq\nr\ns\nt\nu\nv\nw\nd\ne\nf\nx\n",
		"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n",
		"Index: a\n" +
			"--- a/b\n" +
			"+++ b/a\n" +
			"@@ -1,6 +1,9 @@\n" +
			" a\n" +
			" b\n" +
			" c\n" +
			"+d\n" +
			"+e\n" +
			"+f\n" +
			" g\n" +
			" h\n" +
			" i\n" +
			"@@ -8,8 +11,8 @@ j\n" +
			" k\n" +
			" l\n" +
			" m\n" +
			"-N\n" +
			"-O\n" +
			"+n\n" +
			"+o\n" +
			" p\n" +
			" q\n" +
			" r\n" +
			"@@ -18,7 +21,6 @@ t\n" +
			" u\n" +
			" v\n" +
			" w\n" +
			"-d\n" +
			"-e\n" +
			"-f\n" +
			" x\n" +
			"+y\n" +
			"+z\n",
	},
	{
		"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n",
		"",
		"Index: a\n" +
			"deleted file mode 100644\n" +
			"--- a/a\n" +
			"+++ /dev/null\n" +
			"@@ -1,26 +0,0 @@\n" +
			"-a\n" +
			"-b\n" +
			"-c\n" +
			"-d\n" +
			"-e\n" +
			"-f\n" +
			"-g\n" +
			"-h\n" +
			"-i\n" +
			"-j\n" +
			"-k\n" +
			"-l\n" +
			"-m\n" +
			"-n\n" +
			"-o\n" +
			"-p\n" +
			"-q\n" +
			"-r\n" +
			"-s\n" +
			"-t\n" +
			"-u\n" +
			"-v\n" +
			"-w\n" +
			"-x\n" +
			"-y\n" +
			"-z\n",
	},
	{
		"",
		"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n",
		"Index: a\n" +
			"new file mode 100644\n" +
			"--- /dev/null\n" +
			"+++ b/a\n" +
			"@@ -0,0 +1,26 @@\n" +
			"+a\n" +
			"+b\n" +
			"+c\n" +
			"+d\n" +
			"+e\n" +
			"+f\n" +
			"+g\n" +
			"+h\n" +
			"+i\n" +
			"+j\n" +
			"+k\n" +
			"+l\n" +
			"+m\n" +
			"+n\n" +
			"+o\n" +
			"+p\n" +
			"+q\n" +
			"+r\n" +
			"+s\n" +
			"+t\n" +
			"+u\n" +
			"+v\n" +
			"+w\n" +
			"+x\n" +
			"+y\n" +
			"+z\n",
	},
	{
		"\xc2\xd8\xf9\x63\x8c\xf7\xc6\x9b\xb0\x3c\x39\xfa\x08\x8e\x42\x8f" +
			"\x1c\x7c\xaf\x54\x22\x87\xc3\xc5\x68\x9b\xe1\xbd\xbc\xc3\xe0\xda" +
			"\xcc\xe3\x96\xda\xc2\xaf\xbb\x75\x79\x64\x86\x60\x8a\x43\x9e\x07" +
			"\x9c\xaa\x92\x88\xd4\x30\xb9\x8b\x95\x04\x60\x71\xc7\xbb\x2d\x93" +
			"\x66\x73\x01\x24\xf3\x63\xbf\xe6\x1d\x38\x15\x56\x98\xc4\x1f\x85" +
			"\xc3\x60\x39\x3a\x0d\x57\x53\x0c\x29\x3f\xbb\x44\x7e\x56\x56\x9d" +
			"\x87\xcf\xf6\x88\xe8\x98\x05\x85\xf8\xfe\x44\x21\xfa\x33\xc9\xa4" +
			"\x22\xbe\x89\x05\x8b\x82\x76\xc9\x7c\xaf\x48\x28\xc4\x86\x15\x89" +
			"\xb9\x98\xfa\x41\xfc\x3d\x8d\x80\x29\x33\x17\x45\xa5\x7f\x67\x79" +
			"\x7f\x92\x3b\x2e\x4c\xc1\xd2\x1b\x9e\xcf\xed\x53\x56\xb2\x49\x58" +
			"\xd8\xe9\x9f\x98\xa3\xfe\x78\xe1\xe8\x74\x71\x04\x1a\x87\xd9\x68" +
			"\x18\x68\xd0\xae\x7b\xa4\x25\xe3\x06\x03\x7e\x8b\xd3\x50\x1f\xb1" +
			"\x67\x08\xe3\x93\xf4\x4f\xa1\xfb\x31\xcf\x99\x5a\x43\x9f\x4b\xc4" +
			"\xaa\x68\x1a\xf9\x8e\x97\x02\x80\x17\xf1\x25\x21\xdf\x94\xbf\x41" +
			"\x08\x59\x3d\xea\x36\x23\x03\xb5\x62\x4d\xb6\x8f\x9e\xdf\x1f\x03" +
			"\x7d\x70\xe0\x6f\x46\x08\x96\x79\x72\xb7\xae\x41\x2b\xbd\x2a\x95",

		"\x8e\x5f\xf8\x79\x36\x8d\xbe\x68\xc4\x2c\x78\x8a\x46\x28\x40\x3e" +
			"\xcf\x3b\xb9\x14\xaf\xfa\x04\x9e\x4b\xa2\x52\x51\x51\xf0\xad\xd3" +
			"\x03\x1c\x03\x79\x5f\x53\xc7\x1a\xd5\x28\xe2\xd9\x19\x37\xa4\xfa" +
			"\xdd\xff\xac\xb5\xa9\x42\x4e\x17\xeb\xb4\x0d\x20\x67\x08\x43\x21" +
			"\x7d\x12\x27\xfa\x96\x7a\x85\xf8\x04\x5f\xf4\xfe\xda\x9f\x66\xf2" +
			"\xba\x04\x39\x00\xab\x3f\x23\x20\x84\x53\xb4\x88\xb6\xee\xa2\x9e" +
			"\xc1\xca\xd4\x09\x2a\x27\x89\x2f\xcb\xba\xa6\x41\xb6\xe9\xc5\x08" +
			"\xff\xf5\x95\x35\xab\xbb\x5c\x62\x96\xe7\x7c\x8f\xf2\x40\x12\xc9" +
			"\x2d\xfe\xff\x75\x4f\x70\x47\xc9\xcd\x15\x0a\x1c\x23\xe7\x0f\x15" +
			"\x95\x75\x30\x8f\x6e\x9f\x7e\xa5\x9d\xd1\x65\x1c\x4d\x4e\xf4\x32" +
			"\x49\x9b\xa1\x30\x44\x62\x6f\xe2\xe6\x69\x09\xf8\x7c\x7c\xbe\x07" +
			"\xa9\xb6\x14\x7a\x6b\x85\xe4\xbf\x48\xbe\x5b\x3b\x70\xb3\x79\x3b" +
			"\xc4\x35\x9d\x86\xf1\xfe\x2b\x6f\x80\x74\x50\xf3\x96\x59\x53\x1a" +
			"\x75\x46\x9d\x57\x72\xb3\xb1\x26\xf5\x81\xcd\x96\x08\xbc\x2b\x10" +
			"\xdc\x80\xbd\xd0\xdf\x03\x6d\x8d\xec\x30\x2b\x4c\xdb\x4d\x3b\xef" +
			"\x7d\x3a\x39\xc8\x5a\xc4\xcc\x24\x37\xde\xe2\x95\x2b\x04\x97\xb0",

		// From git diff --binary
		"Index: a\n" +
			"index cb34d9b1743b7c410fa750be8a58eb355987110b..0a01764bc1b2fd29da317f72208f462ad342400f 100644\n" +
			"GIT binary patch\n" +
			"literal 256\n" +
			"zcmV+b0ssDvU-)@8jlO8aEO?4WC_p~XJGm6E`UIX!qEb;&@U7DW90Pe@Q^y+BDB{@}\n" +
			"zH>CRA|E#sCLQWU!v<)C<2ty%#5-0kWdWHA|U-bUkpJwv91UUe!KO-Q7Q?!V-?xLQ-\n" +
			"z%G3!eCy6i1x~4(4>BR{D^_4ZNyIf+H=X{UyKoZF<{{MAPa7W3_6$%_9=MNQ?buf=^\n" +
			"zpMIsC(PbP>PV_QKo1rj7VsGN+X$kmze7*;%wiJ46h2+0TzFRwRvw1tjHJyg>{wr^Q\n" +
			"zbWrn_SyLKyMx9r3v#}=ifz6f(yekmgfW6S)18t4$Fe^;kO*`*>IyuN%#LOf&-r|)j\n" +
			"G1edVN^?m&S\n" +
			"\n" +
			"literal 256\n" +
			"zcmV+b0ssEO*!g3O_r{yBJURLZjzW(de6Lg@hr`8ao8i5@!{FM?<CfaOue)`5WQJgh\n" +
			"zL!Jkms*;G*Fu9AB1YmK;yDgJua{(mtW54DdI2Bfy#2<yjU^zMsS5pirKf6SJR#u&d\n" +
			"z&-RGum<5IS{zM`AGs&bPzKI2kf_BM#uSh7wh82mqnEFBdJ&k}VGZ#gre`k4rk~=O;\n" +
			"z!O|O^&+SuIvPoFj>7SUR{&?Z&ba4b4huLTtXwa^Eq$T491AdFsP#>{p2;-CVPoeuU\n" +
			"z&zV|7pG(B5Xd3yBmjZwn@g*VOl)pg;Sv~4DBLlT!O}3Ao-yZ{gaNuu72$p$rx2{1e\n" +
			"Gy(*Pb;D3Ms\n" +
			"\n",
	},
	{
		"\xc2\xd8\xf9\x63\x8c\xf7\xc6\x9b\xb0\x3c\x39\xfa\x08\x8e\x42\x8f" +
			"\x1c\x7c\xaf\x54\x22\x87\xc3\xc5\x68\x9b\xe1\xbd\xbc\xc3\xe0\xda" +
			"\xcc\xe3\x96\xda\xc2\xaf\xbb\x75\x79\x64\x86\x60\x8a\x43\x9e\x07" +
			"\x9c\xaa\x92\x88\xd4\x30\xb9\x8b\x95\x04\x60\x71\xc7\xbb\x2d\x93" +
			"\x66\x73\x01\x24\xf3\x63\xbf\xe6\x1d\x38\x15\x56\x98\xc4\x1f\x85" +
			"\xc3\x60\x39\x3a\x0d\x57\x53\x0c\x29\x3f\xbb\x44\x7e\x56\x56\x9d" +
			"\x87\xcf\xf6\x88\xe8\x98\x05\x85\xf8\xfe\x44\x21\xfa\x33\xc9\xa4" +
			"\x22\xbe\x89\x05\x8b\x82\x76\xc9\x7c\xaf\x48\x28\xc4\x86\x15\x89" +
			"\xb9\x98\xfa\x41\xfc\x3d\x8d\x80\x29\x33\x17\x45\xa5\x7f\x67\x79" +
			"\x7f\x92\x3b\x2e\x4c\xc1\xd2\x1b\x9e\xcf\xed\x53\x56\xb2\x49\x58" +
			"\xd8\xe9\x9f\x98\xa3\xfe\x78\xe1\xe8\x74\x71\x04\x1a\x87\xd9\x68" +
			"\x18\x68\xd0\xae\x7b\xa4\x25\xe3\x06\x03\x7e\x8b\xd3\x50\x1f\xb1" +
			"\x67\x08\xe3\x93\xf4\x4f\xa1\xfb\x31\xcf\x99\x5a\x43\x9f\x4b\xc4" +
			"\xaa\x68\x1a\xf9\x8e\x97\x02\x80\x17\xf1\x25\x21\xdf\x94\xbf\x41" +
			"\x08\x59\x3d\xea\x36\x23\x03\xb5\x62\x4d\xb6\x8f\x9e\xdf\x1f\x03" +
			"\x7d\x70\xe0\x6f\x46\x08\x96\x79\x72\xb7\xae\x41\x2b\xbd\x2a\x95",

		"\x8e\x5f\xf8\x79\x36\x8d\xbe\x68\xc4\x2c\x78\x8a\x46\x28\x40\x3e" +
			"\xcf\x3b\xb9\x14\xaf\xfa\x04\x9e\x4b\xa2\x52\x51\x51\xf0\xad\xd3" +
			"\x03\x1c\x03\x79\x5f\x53\xc7\x1a\xd5\x28\xe2\xd9\x19\x37\xa4\xfa" +
			"\xdd\xff\xac\xb5\xa9\x42\x4e\x17\xeb\xb4\x0d\x20\x67\x08\x43\x21" +
			"\x7d\x12\x27\xfa\x96\x7a\x85\xf8\x04\x5f\xf4\xfe\xda\x9f\x66\xf2" +
			"\xba\x04\x39\x00\xab\x3f\x23\x20\x84\x53\xb4\x88\xb6\xee\xa2\x9e" +
			"\xc1\xca\xd4\x09\x2a\x27\x89\x2f\xcb\xba\xa6\x41\xb6\xe9\xc5\x08" +
			"\xff\xf5\x95\x35\xab\xbb\x5c\x62\x96\xe7\x7c\x8f\xf2\x40\x12\xc9" +
			"\x2d\xfe\xff\x75\x4f\x70\x47\xc9\xcd\x15\x0a\x1c\x23\xe7\x0f\x15" +
			"\x95\x75\x30\x8f\x6e\x9f\x7e\xa5\x9d\xd1\x65\x1c\x4d\x4e\xf4\x32" +
			"\x49\x9b\xa1\x30\x44\x62\x6f\xe2\xe6\x69\x09\xf8\x7c\x7c\xbe\x07" +
			"\xa9\xb6\x14\x7a\x6b\x85\xe4\xbf\x48\xbe\x5b\x3b\x70\xb3\x79\x3b" +
			"\xc4\x35\x9d\x86\xf1\xfe\x2b\x6f\x80\x74\x50\xf3\x96\x59\x53\x1a" +
			"\x75\x46\x9d\x57\x72\xb3\xb1\x26\xf5\x81\xcd\x96\x08\xbc\x2b\x10" +
			"\xdc\x80\xbd\xd0\xdf\x03\x6d\x8d\xec\x30\x2b\x4c\xdb\x4d\x3b\xef" +
			"\x7d\x3a\x39\xc8\x5a\xc4\xcc\x24\x37\xde\xe2\x95\x2b\x04\x97\xb0",

		// From hg diff --git
		"Index: a\n" +
			"index cb34d9b1743b7c410fa750be8a58eb355987110b..0a01764bc1b2fd29da317f72208f462ad342400f\n" +
			"GIT binary patch\n" +
			"literal 256\n" +
			"zc$@(M0ssDvU-)@8jlO8aEO?4WC_p~XJGm6E`UIX!qEb;&@U7DW90Pe@Q^y+BDB{@}\n" +
			"zH>CRA|E#sCLQWU!v<)C<2ty%#5-0kWdWHA|U-bUkpJwv91UUe!KO-Q7Q?!V-?xLQ-\n" +
			"z%G3!eCy6i1x~4(4>BR{D^_4ZNyIf+H=X{UyKoZF<{{MAPa7W3_6$%_9=MNQ?buf=^\n" +
			"zpMIsC(PbP>PV_QKo1rj7VsGN+X$kmze7*;%wiJ46h2+0TzFRwRvw1tjHJyg>{wr^Q\n" +
			"zbWrn_SyLKyMx9r3v#}=ifz6f(yekmgfW6S)18t4$Fe^;kO*`*>IyuN%#LOf&-r|)j\n" +
			"G1edVN^?m&S\n" +
			"\n",
	},
	{
		"",
		"",
		"Index: hello\n" +
			"===================================================================\n" +
			"old mode 100644\n" +
			"new mode 100755\n",
	},
}
