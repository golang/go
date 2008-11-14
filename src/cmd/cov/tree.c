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

// Mutable map structure, but still based on
// Okasaki, Red Black Trees in a Functional Setting, JFP 1999,
// which is a lot easier than the traditional red-black
// and plenty fast enough for me.  (Also I could copy
// and edit fmap.c.)

#include <u.h>
#include <libc.h>
#include "tree.h"

#define TreeNode TreeNode
#define Tree Tree

enum
{
	Red = 0,
	Black = 1
};


// Red-black trees are binary trees with this property:
//	1. No red node has a red parent.
//	2. Every path from the root to a leaf contains the
//		same number of black nodes.

static TreeNode*
rwTreeNode(TreeNode *p, int color, TreeNode *left, void *key, void *value, TreeNode *right)
{
	if(p == nil)
		p = malloc(sizeof *p);
	p->color = color;
	p->left = left;
	p->key = key;
	p->value = value;
	p->right = right;
	return p;
}

static TreeNode*
balance(TreeNode *m0)
{
	void *xk, *xv, *yk, *yv, *zk, *zv;
	TreeNode *a, *b, *c, *d;
	TreeNode *m1, *m2;
	int color;
	TreeNode *left, *right;
	void *key, *value;

	color = m0->color;
	left = m0->left;
	key = m0->key;
	value = m0->value;
	right = m0->right;

	// Okasaki notation: (T is mkTreeNode, B is Black, R is Red, x, y, z are key-value.
	//
	// balance B (T R (T R a x b) y c) z d
	// balance B (T R a x (T R b y c)) z d
	// balance B a x (T R (T R b y c) z d)
	// balance B a x (T R b y (T R c z d))
	//
	//     = T R (T B a x b) y (T B c z d)

	if(color == Black){
		if(left && left->color == Red){
			if(left->left && left->left->color == Red){
				a = left->left->left;
				xk = left->left->key;
				xv = left->left->value;
				b = left->left->right;
				yk = left->key;
				yv = left->value;
				c = left->right;
				zk = key;
				zv = value;
				d = right;
				m1 = left;
				m2 = left->left;
				goto hard;
			}else if(left->right && left->right->color == Red){
				a = left->left;
				xk = left->key;
				xv = left->value;
				b = left->right->left;
				yk = left->right->key;
				yv = left->right->value;
				c = left->right->right;
				zk = key;
				zv = value;
				d = right;
				m1 = left;
				m2 = left->right;
				goto hard;
			}
		}else if(right && right->color == Red){
			if(right->left && right->left->color == Red){
				a = left;
				xk = key;
				xv = value;
				b = right->left->left;
				yk = right->left->key;
				yv = right->left->value;
				c = right->left->right;
				zk = right->key;
				zv = right->value;
				d = right->right;
				m1 = right;
				m2 = right->left;
				goto hard;
			}else if(right->right && right->right->color == Red){
				a = left;
				xk = key;
				xv = value;
				b = right->left;
				yk = right->key;
				yv = right->value;
				c = right->right->left;
				zk = right->right->key;
				zv = right->right->value;
				d = right->right->right;
				m1 = right;
				m2 = right->right;
				goto hard;
			}
		}
	}
	return rwTreeNode(m0, color, left, key, value, right);

hard:
	return rwTreeNode(m0, Red, rwTreeNode(m1, Black, a, xk, xv, b),
		yk, yv, rwTreeNode(m2, Black, c, zk, zv, d));
}

static TreeNode*
ins0(TreeNode *p, void *k, void *v, TreeNode *rw)
{
	if(p == nil)
		return rwTreeNode(rw, Red, nil, k, v, nil);
	if(p->key == k){
		if(rw)
			return rwTreeNode(rw, p->color, p->left, k, v, p->right);
		p->value = v;
		return p;
	}
	if(p->key < k)
		p->left = ins0(p->left, k, v, rw);
	else
		p->right = ins0(p->right, k, v, rw);
	return balance(p);
}

static TreeNode*
ins1(Tree *m, TreeNode *p, void *k, void *v, TreeNode *rw)
{
	int i;

	if(p == nil)
		return rwTreeNode(rw, Red, nil, k, v, nil);
	i = m->cmp(p->key, k);
	if(i == 0){
		if(rw)
			return rwTreeNode(rw, p->color, p->left, k, v, p->right);
		p->value = v;
		return p;
	}
	if(i < 0)
		p->left = ins1(m, p->left, k, v, rw);
	else
		p->right = ins1(m, p->right, k, v, rw);
	return balance(p);
}

void
treeputelem(Tree *m, void *key, void *val, TreeNode *rw)
{
	if(m->cmp)
		m->root = ins1(m, m->root, key, val, rw);
	else
		m->root = ins0(m->root, key, val, rw);
}

void
treeput(Tree *m, void *key, void *val)
{
	treeputelem(m, key, val, nil);
}

void*
treeget(Tree *m, void *key)
{
	int i;
	TreeNode *p;

	p = m->root;
	if(m->cmp){
		for(;;){
			if(p == nil)
				return nil;
			i = m->cmp(p->key, key);
			if(i < 0)
				p = p->left;
			else if(i > 0)
				p = p->right;
			else
				return p->value;
		}
	}else{
		for(;;){
			if(p == nil)
				return nil;
			if(p->key == key)
				return p->value;
			if(p->key < key)
				p = p->left;
			else
				p = p->right;
		}
	}
}
