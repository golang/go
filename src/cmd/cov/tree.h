// Renamed from Map to Tree to avoid conflict with libmach.

/*
Copyright (c) 2003-2007 Russ Cox, Tom Bergan, Austin Clements,
                        Massachusetts Institute of Technology
Portions Copyright (c) 2009 The Go Authors. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

typedef struct Tree Tree;
typedef struct TreeNode TreeNode;
struct Tree
{
        int (*cmp)(void*, void*);
        TreeNode *root;
};

struct TreeNode
{
        int color;
        TreeNode *left;
        void *key;
        void *value;
        TreeNode *right;
};

void *treeget(Tree*, void*);
void treeput(Tree*, void*, void*);
void treeputelem(Tree*, void*, void*, TreeNode*);
