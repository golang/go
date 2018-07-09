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


from __future__ import print_function
import re
import sys

print("Loading Go Runtime support.", file=sys.stderr)
#http://python3porting.com/differences.html
if sys.version > '3':
	xrange = range
# allow to manually reload while developing
goobjfile = gdb.current_objfile() or gdb.objfiles()[0]
goobjfile.pretty_printers = []

# G state (runtime2.go)

def read_runtime_const(varname, default):
  try:
    return int(gdb.parse_and_eval(varname))
  except Exception:
    return int(default)


G_IDLE = read_runtime_const("'runtime._Gidle'", 0)
G_RUNNABLE = read_runtime_const("'runtime._Grunnable'", 1)
G_RUNNING = read_runtime_const("'runtime._Grunning'", 2)
G_SYSCALL = read_runtime_const("'runtime._Gsyscall'", 3)
G_WAITING = read_runtime_const("'runtime._Gwaiting'", 4)
G_MORIBUND_UNUSED = read_runtime_const("'runtime._Gmoribund_unused'", 5)
G_DEAD = read_runtime_const("'runtime._Gdead'", 6)
G_ENQUEUE_UNUSED = read_runtime_const("'runtime._Genqueue_unused'", 7)
G_COPYSTACK = read_runtime_const("'runtime._Gcopystack'", 8)
G_SCAN = read_runtime_const("'runtime._Gscan'", 0x1000)
G_SCANRUNNABLE = G_SCAN+G_RUNNABLE
G_SCANRUNNING = G_SCAN+G_RUNNING
G_SCANSYSCALL = G_SCAN+G_SYSCALL
G_SCANWAITING = G_SCAN+G_WAITING

sts = {
    G_IDLE: 'idle',
    G_RUNNABLE: 'runnable',
    G_RUNNING: 'running',
    G_SYSCALL: 'syscall',
    G_WAITING: 'waiting',
    G_MORIBUND_UNUSED: 'moribund',
    G_DEAD: 'dead',
    G_ENQUEUE_UNUSED: 'enqueue',
    G_COPYSTACK: 'copystack',
    G_SCAN: 'scan',
    G_SCANRUNNABLE: 'runnable+s',
    G_SCANRUNNING: 'running+s',
    G_SCANSYSCALL: 'syscall+s',
    G_SCANWAITING: 'waiting+s',
}


#
#  Value wrappers
#

class SliceValue:
	"Wrapper for slice values."

	def __init__(self, val):
		self.val = val

	@property
	def len(self):
		return int(self.val['len'])

	@property
	def cap(self):
		return int(self.val['cap'])

	def __getitem__(self, i):
		if i < 0 or i >= self.len:
			raise IndexError(i)
		ptr = self.val["array"]
		return (ptr + i).dereference()


#
#  Pretty Printers
#


class StringTypePrinter:
	"Pretty print Go strings."

	pattern = re.compile(r'^struct string( \*)?$')

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
		sval = SliceValue(self.val)
		if sval.len > sval.cap:
			return
		for idx, item in enumerate(sval):
			yield ('[{0}]'.format(idx), item)


class MapTypePrinter:
	"""Pretty print map[K]V types.

	Map-typed go variables are really pointers. dereference them in gdb
	to inspect their contents with this pretty printer.
	"""

	pattern = re.compile(r'^map\[.*\].*$')

	def __init__(self, val):
		self.val = val

	def display_hint(self):
		return 'map'

	def to_string(self):
		return str(self.val.type)

	def children(self):
		B = self.val['B']
		buckets = self.val['buckets']
		oldbuckets = self.val['oldbuckets']
		flags = self.val['flags']
		inttype = self.val['hash0'].type
		cnt = 0
		for bucket in xrange(2 ** int(B)):
			bp = buckets + bucket
			if oldbuckets:
				oldbucket = bucket & (2 ** (B - 1) - 1)
				oldbp = oldbuckets + oldbucket
				oldb = oldbp.dereference()
				if (oldb['overflow'].cast(inttype) & 1) == 0:  # old bucket not evacuated yet
					if bucket >= 2 ** (B - 1):
						continue    # already did old bucket
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
						yield str(cnt), k
						yield str(cnt + 1), v
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
			yield ('[{0}]'.format(i), (ptr + j).dereference())


#
#  Register all the *Printer classes above.
#

def makematcher(klass):
	def matcher(val):
		try:
			if klass.pattern.match(str(val.type)):
				return klass(val)
		except Exception:
			pass
	return matcher

goobjfile.pretty_printers.extend([makematcher(var) for var in vars().values() if hasattr(var, 'pattern')])


#
#  Utilities
#

def pc_to_int(pc):
	# python2 will not cast pc (type void*) to an int cleanly
	# instead python2 and python3 work with the hex string representation
	# of the void pointer which we can parse back into an int.
	# int(pc) will not work.
	try:
		# python3 / newer versions of gdb
		pc = int(pc)
	except gdb.error:
		# str(pc) can return things like
		# "0x429d6c <runtime.gopark+284>", so
		# chop at first space.
		pc = int(str(pc).split(None, 1)[0], 16)
	return pc


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
		return str(val['tab'].type) == "struct runtime.itab *" and str(val['data'].type) == "void *"
	except gdb.error:
		pass


def is_eface(val):
	try:
		return str(val['_type'].type) == "struct runtime._type *" and str(val['data'].type) == "void *"
	except gdb.error:
		pass


def lookup_type(name):
	try:
		return gdb.lookup_type(name)
	except gdb.error:
		pass
	try:
		return gdb.lookup_type('struct ' + name)
	except gdb.error:
		pass
	try:
		return gdb.lookup_type('struct ' + name[1:]).pointer()
	except gdb.error:
		pass


def iface_commontype(obj):
	if is_iface(obj):
		go_type_ptr = obj['tab']['_type']
	elif is_eface(obj):
		go_type_ptr = obj['_type']
	else:
		return

	return go_type_ptr.cast(gdb.lookup_type("struct reflect.rtype").pointer()).dereference()


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
		except Exception:
			return "<bad dynamic type>"

		if dtype is None:  # trouble looking up, print something reasonable
			return "({typename}){data}".format(
				typename=iface_dtype_name(self.val), data=self.val['data'])

		try:
			return self.val['data'].cast(dtype).dereference()
		except Exception:
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

	how = ((StringTypePrinter, 'len'), (SliceTypePrinter, 'len'), (MapTypePrinter, 'count'), (ChanTypePrinter, 'qcount'))

	def __init__(self):
		gdb.Function.__init__(self, "len")

	def invoke(self, obj):
		typename = str(obj.type)
		for klass, fld in self.how:
			if klass.pattern.match(typename):
				return obj[fld]


class GoCapFunc(gdb.Function):
	"Capacity of slices or channels"

	how = ((SliceTypePrinter, 'cap'), (ChanTypePrinter, 'dataqsiz'))

	def __init__(self):
		gdb.Function.__init__(self, "cap")

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
		gdb.Function.__init__(self, "dtype")

	def invoke(self, obj):
		try:
			return obj['data'].cast(iface_dtype(obj))
		except gdb.error:
			pass
		return obj

#
#  Commands
#

def linked_list(ptr, linkfield):
	while ptr:
		yield ptr
		ptr = ptr[linkfield]


class GoroutinesCmd(gdb.Command):
	"List all goroutines."

	def __init__(self):
		gdb.Command.__init__(self, "info goroutines", gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

	def invoke(self, _arg, _from_tty):
		# args = gdb.string_to_argv(arg)
		vp = gdb.lookup_type('void').pointer()
		for ptr in SliceValue(gdb.parse_and_eval("'runtime.allgs'")):
			if ptr['atomicstatus'] == G_DEAD:
				continue
			s = ' '
			if ptr['m']:
				s = '*'
			pc = ptr['sched']['pc'].cast(vp)
			pc = pc_to_int(pc)
			blk = gdb.block_for_pc(pc)
			status = int(ptr['atomicstatus'])
			st = sts.get(status, "unknown(%d)" % status)
			print(s, ptr['goid'], "{0:8s}".format(st), blk.function)


def find_goroutine(goid):
	"""
	find_goroutine attempts to find the goroutine identified by goid.
	It returns a tuple of gdb.Value's representing the stack pointer
	and program counter pointer for the goroutine.

	@param int goid

	@return tuple (gdb.Value, gdb.Value)
	"""
	vp = gdb.lookup_type('void').pointer()
	for ptr in SliceValue(gdb.parse_and_eval("'runtime.allgs'")):
		if ptr['atomicstatus'] == G_DEAD:
			continue
		if ptr['goid'] == goid:
			break
	else:
		return None, None
	# Get the goroutine's saved state.
	pc, sp = ptr['sched']['pc'], ptr['sched']['sp']
	status = ptr['atomicstatus']&~G_SCAN
	# Goroutine is not running nor in syscall, so use the info in goroutine
	if status != G_RUNNING and status != G_SYSCALL:
		return pc.cast(vp), sp.cast(vp)

	# If the goroutine is in a syscall, use syscallpc/sp.
	pc, sp = ptr['syscallpc'], ptr['syscallsp']
	if sp != 0:
		return pc.cast(vp), sp.cast(vp)
	# Otherwise, the goroutine is running, so it doesn't have
	# saved scheduler state. Find G's OS thread.
	m = ptr['m']
	if m == 0:
		return None, None
	for thr in gdb.selected_inferior().threads():
		if thr.ptid[1] == m['procid']:
			break
	else:
		return None, None
	# Get scheduler state from the G's OS thread state.
	curthr = gdb.selected_thread()
	try:
		thr.switch()
		pc = gdb.parse_and_eval('$pc')
		sp = gdb.parse_and_eval('$sp')
	finally:
		curthr.switch()
	return pc.cast(vp), sp.cast(vp)


class GoroutineCmd(gdb.Command):
	"""Execute gdb command in the context of goroutine <goid>.

	Switch PC and SP to the ones in the goroutine's G structure,
	execute an arbitrary gdb command, and restore PC and SP.

	Usage: (gdb) goroutine <goid> <gdbcmd>

	You could pass "all" as <goid> to apply <gdbcmd> to all goroutines.

	For example: (gdb) goroutine all <gdbcmd>

	Note that it is ill-defined to modify state in the context of a goroutine.
	Restrict yourself to inspecting values.
	"""

	def __init__(self):
		gdb.Command.__init__(self, "goroutine", gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

	def invoke(self, arg, _from_tty):
		goid_str, cmd = arg.split(None, 1)
		goids = []

		if goid_str == 'all':
			for ptr in SliceValue(gdb.parse_and_eval("'runtime.allgs'")):
				goids.append(int(ptr['goid']))
		else:
			goids = [int(gdb.parse_and_eval(goid_str))]

		for goid in goids:
			self.invoke_per_goid(goid, cmd)

	def invoke_per_goid(self, goid, cmd):
		pc, sp = find_goroutine(goid)
		if not pc:
			print("No such goroutine: ", goid)
			return
		pc = pc_to_int(pc)
		save_frame = gdb.selected_frame()
		gdb.parse_and_eval('$save_sp = $sp')
		gdb.parse_and_eval('$save_pc = $pc')
		# In GDB, assignments to sp must be done from the
		# top-most frame, so select frame 0 first.
		gdb.execute('select-frame 0')
		gdb.parse_and_eval('$sp = {0}'.format(str(sp)))
		gdb.parse_and_eval('$pc = {0}'.format(str(pc)))
		try:
			gdb.execute(cmd)
		finally:
			# In GDB, assignments to sp must be done from the
			# top-most frame, so select frame 0 first.
			gdb.execute('select-frame 0')
			gdb.parse_and_eval('$pc = $save_pc')
			gdb.parse_and_eval('$sp = $save_sp')
			save_frame.select()


class GoIfaceCmd(gdb.Command):
	"Print Static and dynamic interface types"

	def __init__(self):
		gdb.Command.__init__(self, "iface", gdb.COMMAND_DATA, gdb.COMPLETE_SYMBOL)

	def invoke(self, arg, _from_tty):
		for obj in gdb.string_to_argv(arg):
			try:
				#TODO fix quoting for qualified variable names
				obj = gdb.parse_and_eval(str(obj))
			except Exception as e:
				print("Can't parse ", obj, ": ", e)
				continue

			if obj['data'] == 0:
				dtype = "nil"
			else:
				dtype = iface_dtype(obj)

			if dtype is None:
				print("Not an interface: ", obj.type)
				continue

			print("{0}: {1}".format(obj.type, dtype))

# TODO: print interface's methods and dynamic type's func pointers thereof.
#rsc: "to find the number of entries in the itab's Fn field look at
# itab.inter->numMethods
# i am sure i have the names wrong but look at the interface type
# and its method count"
# so Itype will start with a commontype which has kind = interface

#
# Register all convenience functions and CLI commands
#
GoLenFunc()
GoCapFunc()
DTypeFunc()
GoroutinesCmd()
GoroutineCmd()
GoIfaceCmd()
