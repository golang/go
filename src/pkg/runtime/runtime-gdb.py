# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""GDB Pretty printers and convencience functions for Go's runtime structures.

This script is loaded by GDB when it finds a .debug_gdb_scripts
section in the compiled binary.  The [68]l linkers emit this with a
path to this file based on the path to the runtime package.
"""

import sys, re

print >>sys.stderr, "Loading Go Runtime support."

#
#  Pretty Printers
#

class StringTypePrinter:
	"Pretty print Go strings."

	pattern = re.compile(r'^struct string$')

	def __init__(self, val):
		self.val = val

	def display_hint(self):
		return 'string'

	def to_string(self):
		return self.val['str']


class SliceTypePrinter:
	"Pretty print slices."

	pattern = re.compile(r'^struct \[\]')

	def __init__(self, val):
		self.val = val

	def display_hint(self):
		return 'array'

	def to_string(self):
		return str(self.val.type)[6:]  # skip 'struct '

	def children(self):
		ptr = self.val["array"]
		for idx in range(self.val["len"]):
			yield ('[%d]' % idx, (ptr + idx).dereference())


class MapTypePrinter:
	"""Pretty print map[K]V types.

	Map-typed go variables are really pointers. dereference them in gdb
	to inspect their contents with this pretty printer.
	"""

	pattern = re.compile(r'^struct hash<.*>$')

	def __init__(self, val):
		self.val = val

	def display_hint(self):
		return 'map'

	def to_string(self):
		return str(self.val.type)

	def children(self):
		stab = self.val['st']
		i = 0
		for v in self.traverse_hash(stab):
			yield ("[%d]" %  i, v['key'])
			yield ("[%d]" % (i + 1), v['val'])
			i += 2

	def traverse_hash(self, stab):
		ptr = stab['entry'].address
		end = stab['end']
		while ptr < end:
			v = ptr.dereference()
			ptr = ptr + 1
			if v['hash'] == 0: continue
			if v['hash'] & 63 == 63:   # subtable
				for v in self.traverse_hash(v['key'].cast(self.val['st'].type)):
					yield v
			else:
				yield v


class ChanTypePrinter:
	"""Pretty print chan[T] types.

	Map-typed go variables are really pointers. dereference them in gdb
	to inspect their contents with this pretty printer.
	"""

	pattern = re.compile(r'^struct hchan<.*>$')

	def __init__(self, val):
		self.val = val

	def display_hint(self):
		return 'array'

	def to_string(self):
		return str(self.val.type)

	def children(self):
		ptr = self.val['recvdataq']
		for idx in range(self.val["qcount"]):
			yield ('[%d]' % idx, ptr['elem'])
			ptr = ptr['link']

#
#  Register all the *Printer classes
#

def makematcher(klass):
	def matcher(val):
		try:
			if klass.pattern.match(str(val.type)): return klass(val)
		except: pass
	return matcher

gdb.current_objfile().pretty_printers.extend([makematcher(k) for k in vars().values() if hasattr(k, 'pattern')])


#
#  Convenience Functions
#

class GoLenFunc(gdb.Function):
	"Length of strings, slices, maps or channels"

        how = ((StringTypePrinter, 'len' ),
               (SliceTypePrinter, 'len'),
               (MapTypePrinter, 'count'),
               (ChanTypePrinter, 'qcount'))

	def __init__(self):
		super(GoLenFunc, self).__init__("len")

	def invoke(self, obj):
		typename = str(obj.type)
                for klass, fld in self.how:
			if klass.pattern.match(typename):
				return obj[fld]

class GoCapFunc(gdb.Function):
	"Capacity of slices or channels"

        how = ((SliceTypePrinter, 'cap'),
               (ChanTypePrinter, 'dataqsiz'))

	def __init__(self):
		super(GoCapFunc, self).__init__("cap")

	def invoke(self, obj):
		typename = str(obj.type)
                for klass, fld in self.how:
			if klass.pattern.match(typename):
				return obj[fld]

#
# Register all convience functions and CLI commands
#
for k in vars().values():
	if hasattr(k, 'invoke'):
		k()
