# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""GDB Pretty printers and convenience functions for Go's runtime structures.

This script is loaded by GDB when it finds a .debug_gdb_scripts
section in the compiled binary. The [68]l linkers emit this with a
path to this file based on the path to the runtime package.
"""

# Known issues:
#    - pretty printing only works for the 'native' strings. E.g. 'type
#      foo string' will make foo a plain struct in the eyes of gdb,
#      circumventing the pretty print triggering.


import sys, re

print >>sys.stderr, "Loading Go Runtime support."

# allow to manually reload while developing
goobjfile = gdb.current_objfile() or gdb.objfiles()[0]
goobjfile.pretty_printers = []

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
		l = int(self.val['len'])
		return self.val['str'].string("utf-8", "ignore", l)


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
		if self.val["len"] > self.val["cap"]:
			return
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
		B = self.val['b']
		buckets = self.val['buckets']
		oldbuckets = self.val['oldbuckets']
		flags = self.val['flags']
		inttype = self.val['hash0'].type
		cnt = 0
		for bucket in xrange(2 ** B):
			bp = buckets + bucket
			if oldbuckets:
				oldbucket = bucket & (2 ** (B - 1) - 1)
				oldbp = oldbuckets + oldbucket
				oldb = oldbp.dereference()
				if (oldb['overflow'].cast(inttype) & 1) == 0: # old bucket not evacuated yet
					if bucket >= 2 ** (B - 1): continue   # already did old bucket
					bp = oldbp
			while bp:
				b = bp.dereference()
				for i in xrange(8):
					if b['tophash'][i] != 0:
						k = b['keys'][i]
						v = b['values'][i]
						if flags & 1:
							k = k.dereference()
						if flags & 2:
							v = v.dereference()
						yield '%d' % cnt, k
						yield '%d' % (cnt + 1), v
						cnt += 2
				bp = b['overflow']

class ChanTypePrinter:
	"""Pretty print chan[T] types.

	Chan-typed go variables are really pointers. dereference them in gdb
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
		# see chan.c chanbuf(). et is the type stolen from hchan<T>::recvq->first->elem
		et = [x.type for x in self.val['recvq']['first'].type.target().fields() if x.name == 'elem'][0]
		ptr = (self.val.address + 1).cast(et.pointer())
		for i in range(self.val["qcount"]):
			j = (self.val["recvx"] + i) % self.val["dataqsiz"]
			yield ('[%d]' % i, (ptr + j).dereference())


#
#  Register all the *Printer classes above.
#

def makematcher(klass):
	def matcher(val):
		try:
			if klass.pattern.match(str(val.type)):
				return klass(val)
		except:
			pass
	return matcher

goobjfile.pretty_printers.extend([makematcher(k) for k in vars().values() if hasattr(k, 'pattern')])

#
#  For reference, this is what we're trying to do:
#  eface: p *(*(struct 'runtime.rtype'*)'main.e'->type_->data)->string
#  iface: p *(*(struct 'runtime.rtype'*)'main.s'->tab->Type->data)->string
#
# interface types can't be recognized by their name, instead we check
# if they have the expected fields.  Unfortunately the mapping of
# fields to python attributes in gdb.py isn't complete: you can't test
# for presence other than by trapping.


def is_iface(val):
	try:
		return str(val['tab'].type) == "struct runtime.itab *" \
		      and str(val['data'].type) == "void *"
	except:
		pass

def is_eface(val):
	try:
		return str(val['_type'].type) == "struct runtime._type *" \
		      and str(val['data'].type) == "void *"
	except:
		pass

def lookup_type(name):
	try:
		return gdb.lookup_type(name)
	except:
		pass
	try:
		return gdb.lookup_type('struct ' + name)
	except:
		pass
	try:
		return gdb.lookup_type('struct ' + name[1:]).pointer()
	except:
		pass

_rctp_type = gdb.lookup_type("struct runtime.rtype").pointer()

def iface_commontype(obj):
	if is_iface(obj):
		go_type_ptr = obj['tab']['_type']
	elif is_eface(obj):
		go_type_ptr = obj['_type']
	else:
		return
	
	return go_type_ptr.cast(_rctp_type).dereference()
	

def iface_dtype(obj):
	"Decode type of the data field of an eface or iface struct."
	# known issue: dtype_name decoded from runtime.rtype is "nested.Foo"
	# but the dwarf table lists it as "full/path/to/nested.Foo"

	dynamic_go_type = iface_commontype(obj)
	if dynamic_go_type is None:
		return
	dtype_name = dynamic_go_type['string'].dereference()['str'].string()

	dynamic_gdb_type = lookup_type(dtype_name)
	if dynamic_gdb_type is None:
		return
	
	type_size = int(dynamic_go_type['size'])
	uintptr_size = int(dynamic_go_type['size'].type.sizeof)	 # size is itself an uintptr
	if type_size > uintptr_size:
			dynamic_gdb_type = dynamic_gdb_type.pointer()

	return dynamic_gdb_type

def iface_dtype_name(obj):
	"Decode type name of the data field of an eface or iface struct."

	dynamic_go_type = iface_commontype(obj)
	if dynamic_go_type is None:
		return
	return dynamic_go_type['string'].dereference()['str'].string()


class IfacePrinter:
	"""Pretty print interface values

	Casts the data field to the appropriate dynamic type."""

	def __init__(self, val):
		self.val = val

	def display_hint(self):
		return 'string'

	def to_string(self):
		if self.val['data'] == 0:
			return 0x0
		try:
			dtype = iface_dtype(self.val)
		except:
			return "<bad dynamic type>"

		if dtype is None:  # trouble looking up, print something reasonable
			return "(%s)%s" % (iface_dtype_name(self.val), self.val['data'])

		try:
			return self.val['data'].cast(dtype).dereference()
		except:
			pass
		return self.val['data'].cast(dtype)


def ifacematcher(val):
	if is_iface(val) or is_eface(val):
		return IfacePrinter(val)

goobjfile.pretty_printers.append(ifacematcher)

#
#  Convenience Functions
#

class GoLenFunc(gdb.Function):
	"Length of strings, slices, maps or channels"

	how = ((StringTypePrinter, 'len'),
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

class DTypeFunc(gdb.Function):
	"""Cast Interface values to their dynamic type.

	For non-interface types this behaves as the identity operation.
	"""

	def __init__(self):
		super(DTypeFunc, self).__init__("dtype")

	def invoke(self, obj):
		try:
			return obj['data'].cast(iface_dtype(obj))
		except:
			pass
		return obj

#
#  Commands
#

sts = ('idle', 'runnable', 'running', 'syscall', 'waiting', 'moribund', 'dead', 'recovery')

def linked_list(ptr, linkfield):
	while ptr:
		yield ptr
		ptr = ptr[linkfield]


class GoroutinesCmd(gdb.Command):
	"List all goroutines."

	def __init__(self):
		super(GoroutinesCmd, self).__init__("info goroutines", gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

	def invoke(self, arg, from_tty):
		# args = gdb.string_to_argv(arg)
		vp = gdb.lookup_type('void').pointer()
		for ptr in linked_list(gdb.parse_and_eval("'runtime.allg'"), 'alllink'):
			if ptr['status'] == 6:	# 'gdead'
				continue
			s = ' '
			if ptr['m']:
				s = '*'
			pc = ptr['sched']['pc'].cast(vp)
			sp = ptr['sched']['sp'].cast(vp)
			blk = gdb.block_for_pc(long((pc)))
			print s, ptr['goid'], "%8s" % sts[long((ptr['status']))], blk.function

def find_goroutine(goid):
	vp = gdb.lookup_type('void').pointer()
	for ptr in linked_list(gdb.parse_and_eval("'runtime.allg'"), 'alllink'):
		if ptr['status'] == 6:	# 'gdead'
			continue
		if ptr['goid'] == goid:
			return [ptr['sched'][x].cast(vp) for x in 'pc', 'sp']
	return None, None


class GoroutineCmd(gdb.Command):
	"""Execute gdb command in the context of goroutine <goid>.

	Switch PC and SP to the ones in the goroutine's G structure,
	execute an arbitrary gdb command, and restore PC and SP.

	Usage: (gdb) goroutine <goid> <gdbcmd>

	Note that it is ill-defined to modify state in the context of a goroutine.
	Restrict yourself to inspecting values.
	"""

	def __init__(self):
		super(GoroutineCmd, self).__init__("goroutine", gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

	def invoke(self, arg, from_tty):
		goid, cmd = arg.split(None, 1)
		goid = gdb.parse_and_eval(goid)
		pc, sp = find_goroutine(int(goid))
		if not pc:
			print "No such goroutine: ", goid
			return
		save_frame = gdb.selected_frame()
		gdb.parse_and_eval('$save_pc = $pc')
		gdb.parse_and_eval('$save_sp = $sp')
		gdb.parse_and_eval('$pc = 0x%x' % long(pc))
		gdb.parse_and_eval('$sp = 0x%x' % long(sp))
		try:
			gdb.execute(cmd)
		finally:
			gdb.parse_and_eval('$pc = $save_pc')
			gdb.parse_and_eval('$sp = $save_sp')
			save_frame.select()


class GoIfaceCmd(gdb.Command):
	"Print Static and dynamic interface types"

	def __init__(self):
		super(GoIfaceCmd, self).__init__("iface", gdb.COMMAND_DATA, gdb.COMPLETE_SYMBOL)

	def invoke(self, arg, from_tty):
		for obj in gdb.string_to_argv(arg):
			try:
				#TODO fix quoting for qualified variable names
				obj = gdb.parse_and_eval("%s" % obj)
			except Exception, e:
				print "Can't parse ", obj, ": ", e
				continue

			if obj['data'] == 0:
				dtype = "nil"
			else:
				dtype = iface_dtype(obj)
				
			if dtype is None:
				print "Not an interface: ", obj.type
				continue

			print "%s: %s" % (obj.type, dtype)

# TODO: print interface's methods and dynamic type's func pointers thereof.
#rsc: "to find the number of entries in the itab's Fn field look at itab.inter->numMethods
#i am sure i have the names wrong but look at the interface type and its method count"
# so Itype will start with a commontype which has kind = interface

#
# Register all convenience functions and CLI commands
#
for k in vars().values():
	if hasattr(k, 'invoke'):
		k()
