#!/usr/bin/env python
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from xml.etree.cElementTree import *
from os.path import basename, exists
import getopt
import sys
import re

_ns = None

outfile = None
golines = []
def go(fmt, *args):
	golines.append(fmt % args)

namere = re.compile('([A-Z0-9][a-z]+|[A-Z0-9]+(?![a-z])|[a-z]+)')
allcaps = re.compile('^[A-Z0-9]+$')

sizeoftab = {
	"byte": 1,
	"int8": 1,
	"uint8": 1,
	"int16": 2,
	"uint16": 2,
	"int32": 4,
	"uint32": 4,
	"float32": 4,
	"float64": 8,
	"Id": 4,
	"Keysym": 4,
	"Timestamp": 4,
}

def sizeof(t):
	if t in sizeoftab:
		return sizeoftab[t]
	return 4

symbols = []

def readsymbols(filename):
	symbols.append("XXX Dummy XXX")
	if exists(filename):
		for line in open(filename, 'r').readlines():
			symbols.append(line.strip())

#
# Name munging crap for names, enums and types.
#

mangletab = {
	"int8_t":	"int8",
	"uint8_t":	"byte",
	"uint16_t":	"uint16",
	"uint32_t":	"uint32",
	"int16_t":	"int16",
	"int32_t":	"int32",
	"float":		"float32",
	"double":	"float64",
	"char":		"byte",
	"void":		"byte",
	'VISUALTYPE':	'VisualInfo',
	'DEPTH':		'DepthInfo',
	'SCREEN':	'ScreenInfo',
	'Setup':		'SetupInfo',
	'WINDOW':	'Id',
}

def mangle(str):
	if str in mangletab:
		return mangletab[str]
	return str

def camel(str):
	return str[0].upper() + str[1:]
def uncamel(str):
	return str[0].lower() + str[1:]

def nitem(str):
	split = namere.finditer(str)
	return ''.join([camel(match.group(0)) for match in split])

def titem(str):
	str = mangle(str)
	if str in sizeoftab:
		return str
	if allcaps.match(str):
		return str.capitalize()
	return nitem(str)

def n(list):
	"Mangle name (JoinedCamelCase) and chop off 'xcb' prefix."
	if len(list) == 1:
		parts = [nitem(list[0])]
	else:
		parts = [nitem(x) for x in list[1:]]
	return ''.join(parts)

def t(list):
	"Mangle name (JoinedCamelCase) and chop off 'xcb' prefix. Preserve primitive type names."
	if len(list) == 1:
		return titem(list[0])
	else:
		parts = [titem(x) for x in list[1:]]
	return ''.join(parts)

#
# Various helper functions
#

def go_type_setup(self, name, postfix):
	'''
	Sets up all the Go-related state by adding additional data fields to
	all Field and Type objects. Here is where we figure out most of our
	variable and function names.

	Recurses into child fields and list member types.
	'''
	# Do all the various names in advance
	self.c_type = t(name + postfix)
	self.c_request_name = n(name)
	self.c_reply_name = n(name + ('Reply',))
	self.c_reply_type = t(name + ('Reply',))

	if not self.is_container:
		return
	
	offset = 0
	for field in self.fields:
		go_type_setup(field.type, field.field_type, ())
		if field.type.is_list:
			go_type_setup(field.type.member, field.field_type, ())
		field.c_field_type = t(field.field_type)
		field.c_field_name = n((field.field_name,))
		field.c_subscript = '[%d]' % field.type.nmemb if (field.type.nmemb > 1) else ''
		field.c_pointer = ' ' if field.type.nmemb == 1 else '[]'
		field.c_offset = offset
		if field.type.fixed_size():
			offset += field.type.size * field.type.nmemb

def go_accessor_length(expr, prefix, iswriting):
	'''
	Figures out what C code is needed to get a length field.
	For fields that follow a variable-length field, use the accessor.
	Otherwise, just reference the structure field directly.
	'''
	prefarrow = '' if prefix == '' else prefix + '.'
	if expr.lenfield_name != None:
		lenstr = prefarrow + n((expr.lenfield_name,))
		if iswriting and lenstr.endswith("Len"):
			# chop off ...Len and refer to len(array) instead
			return "len(" +   lenstr[:-3] + ")"
		return "int(" + lenstr + ")"
	else:
		return str(expr.nmemb)

def go_accessor_expr(expr, prefix, iswriting):
	'''
	Figures out what C code is needed to get the length of a list field.
	Recurses for math operations.
	Returns bitcount for value-mask fields.
	Otherwise, uses the value of the length field.
	'''
	lenexp = go_accessor_length(expr, prefix, iswriting)
	if expr.op != None:
		return '(' + go_accessor_expr(expr.lhs, prefix, iswriting) + ' ' + expr.op + ' ' + go_accessor_expr(expr.rhs, prefix, iswriting) + ')'
	elif expr.bitfield:
		return 'popCount(' + lenexp + ')'
	else:
		return lenexp

def go_complex(self, fieldlist=None):
	'''
	Helper function for handling all structure types.
	Called for all structs, requests, replies, events, errors.
	'''
	if self.is_union:
		go('type %s struct /*union */ {', self.c_type)
	else:
		go('type %s struct {', self.c_type)
	if not fieldlist:
		fieldlist = self.fields
	for field in fieldlist:
		if field.type.is_pad:
			continue
		if field.wire and field.type.fixed_size():
			go('	%s %s%s', field.c_field_name, field.c_subscript, field.c_field_type)
		if field.wire and not field.type.fixed_size():
			go('	%s []%s', field.c_field_name, field.c_field_type)
	go('}')
	go('')

def go_get(dst, ofs, typename, typesize):
	dst = "v." + dst
	if typesize == 1:
		if typename == 'byte':
			go('%s = b[%s]', dst, ofs)
		else:
			go('%s = %s(b[%s])', dst, typename, ofs)
	elif typesize == 2:
		if typename == 'uint16':
			go('%s = get16(b[%s:])', dst, ofs)
		else:
			go('%s = %s(get16(b[%s:]))', dst, typename, ofs)
	elif typesize == 4:
		if typename == 'uint32':
			go('%s = get32(b[%s:])', dst, ofs)
		else:
			go('%s = %s(get32(b[%s:]))', dst, typename, ofs)
	else:
		go('get%s(b[%s:], &%s)', typename, ofs, dst)

def go_get_list(dst, ofs, typename, typesize, count):
	if typesize == 1 and typename == 'byte':
		go('copy(v.%s[0:%s], b[%s:])', dst, count, ofs)
	else:
		go('for i := 0; i < %s; i++ {', count)
		go_get(dst + "[i]", ofs + "+i*" + str(typesize), typename, typesize)
		go('}')


def go_complex_reader_help(self, fieldlist):
	firstvar = 1
	total = 0
	for field in fieldlist:
		fieldname = field.c_field_name
		fieldtype = field.c_field_type
		if field.wire and field.type.fixed_size():
			total = field.c_offset + field.type.size * field.type.nmemb
			if field.type.is_pad:
				continue
			if field.type.nmemb == 1:
				go_get(fieldname, field.c_offset, fieldtype, field.type.size)
			else:
				go_get_list(fieldname, field.c_offset, fieldtype, field.type.size, field.type.nmemb)
		if field.wire and not field.type.fixed_size():
			lenstr = go_accessor_expr(field.type.expr, 'v', False)
			if firstvar:
				firstvar = 0
				go('offset := %d', field.c_offset)
			else:
				go('offset = pad(offset)')
			go('v.%s = make([]%s, %s)', fieldname, fieldtype, lenstr)
			if fieldtype in sizeoftab:
				go_get_list(fieldname, "offset", fieldtype, sizeoftab[fieldtype], "len(v."+fieldname+")")
				go('offset += len(v.%s) * %d', fieldname, sizeoftab[fieldtype])
			else:
				go('for i := 0; i < %s; i++ {', lenstr)
				go('	offset += get%s(b[offset:], &v.%s[i])', fieldtype, fieldname)
				go('}')
	if not firstvar:
		return 'offset'
	return str(total)

def go_complex_reader(self):
	go('func get%s(b []byte, v *%s) int {', self.c_type, self.c_type)
	go('	return %s', go_complex_reader_help(self, self.fields))
	go('}')
	go('')
	
def structsize(fieldlist):
	fixedtotal = 0
	for field in fieldlist:
		if field.wire and field.type.fixed_size():
			fixedtotal += field.type.size * field.type.nmemb
	return fixedtotal

def go_put(src, ofs, typename, typesize):
	if typesize == 1:
		if typename == 'byte':
			go('b[%s] = %s', ofs, src)
		else:
			go('b[%s] = byte(%s)', ofs, src)
	elif typesize == 2:
		if typename == 'uint16':
			go('put16(b[%s:], %s)', ofs, src)
		else:
			go('put16(b[%s:], uint16(%s))', ofs, src)
	elif typesize == 4:
		if typename == 'uint32':
			go('put32(b[%s:], %s)', ofs, src)
		else:
			go('put32(b[%s:], uint32(%s))', ofs, src)
	else:
		go('put%s(b[%s:], %s)', typename, ofs, src)


def go_complex_writer_help(fieldlist, prefix=''):
	prefarrow = '' if prefix == '' else prefix + '.'
	for field in fieldlist:
		fieldname = prefarrow + field.c_field_name
		fieldtype = field.c_field_type
		if fieldname.endswith("Len"):
			fieldname = "len(%s)" % fieldname[:-3]
			fieldtype = "(exp)"
		if not field.type.fixed_size():
			continue
		if field.type.is_expr:
			expstr = go_accessor_expr(field.type.expr, prefix, True)
			go_put(expstr, field.c_offset, "(exp)", field.type.size)
		elif not field.type.is_pad:
			if field.type.nmemb == 1:
				go_put(fieldname, field.c_offset, fieldtype, field.type.size)
			else:
				go('	copy(b[%d:%d], %s)', field.c_offset, field.c_offset + field.type.nmemb, fieldname)

def go_complex_writer_arguments(param_fields, endstr):
	out = []
	for field in param_fields:
		namestr = field.c_field_name
		typestr = field.c_pointer + t(field.field_type)
		if typestr == '[]byte' and namestr == 'Name':
			typestr = 'string'
		out.append(namestr + ' ' + typestr)
	go('	' + ', '.join(out) + ')' + endstr)

def go_complex_writer_arguments_names(param_fields):
	out = []
	for field in param_fields:
		out.append(field.c_field_name)
	return ', '.join(out)

def go_complex_writer(self, name, void):
	func_name = self.c_request_name

	param_fields = []
	wire_fields = []
	for field in self.fields:
		if field.visible:
			# _len is taken from the list directly
			if not field.field_name.endswith("_len"):
				# The field should appear as a call parameter
				param_fields.append(field)
		if field.wire and not field.auto:
			# We need to set the field up in the structure
			wire_fields.append(field)
	
	if void:
		go('func (c *Conn) %s(', func_name)
		go_complex_writer_arguments(param_fields, "{")
	else:
		go('func (c *Conn) %sRequest(', func_name)
		go_complex_writer_arguments(param_fields, "Cookie {")
	
	fixedtotal = structsize(self.fields)
	if fixedtotal <= 32:
		go('	b := c.scratch[0:%d]', fixedtotal)
	else:
		go('	b := make([]byte, %d)', fixedtotal)
	firstvar = 0
	for field in wire_fields:
		if not field.type.fixed_size():
			if not firstvar:
				firstvar = 1
				go('	n := %d', fixedtotal)
			go('	n += pad(%s * %d)', go_accessor_expr(field.type.expr, '', True), field.type.size)
	if not firstvar:
		go('	put16(b[2:], %d)', fixedtotal / 4)
	else:
		go('	put16(b[2:], uint16(n / 4))')
	go('	b[0] = %s', self.opcode)
	go_complex_writer_help(wire_fields)
	if not void:
		if firstvar:
			go('	cookie := c.sendRequest(b)')
		else:
			go('	return c.sendRequest(b)')
	else:
		go('	c.sendRequest(b)')
	
	# send extra data
	for field in param_fields:
		if not field.type.fixed_size():
			if field.type.is_list:
				fieldname = field.c_field_name
				lenstr = go_accessor_expr(field.type.expr, '', True)
				if t(field.field_type) == 'byte':
					if fieldname == 'Name':
						go('	c.sendString(%s)', fieldname)
					else:
						go('	c.sendBytes(%s[0:%s])', fieldname, lenstr)
				elif t(field.field_type) == 'uint32':
					go('	c.sendUInt32List(%s[0:%s])', fieldname, lenstr)
				else:
					go('	c.send%sList(%s, %s)', t(field.field_type), fieldname, lenstr)
	
	if not void and firstvar:
		go('	return cookie')
	go('}')
	go('')
	
	if not void:
		args = go_complex_writer_arguments_names(param_fields)
		go('func (c *Conn) %s(', func_name)
		go_complex_writer_arguments(param_fields, '(*%s, os.Error) {' % self.c_reply_type)
		go('	return c.%sReply(c.%sRequest(%s))', func_name, func_name, args)
		go('}')
		go('')

#
# Struct definitions, readers and writers
#

def go_struct(self, name):
	go_type_setup(self, name, ())
	if symbols and t(name) not in symbols:
		go('// excluding struct %s\n', t(name))
		return
	
	if self.c_type == 'SetupRequest': return
	if self.c_type == 'SetupFailed': return
	if self.c_type == 'SetupAuthenticate': return

	go_complex(self)
	go_complex_reader(self)
	
	if self.c_type == 'Format': return
	if self.c_type == 'VisualInfo': return
	if self.c_type == 'DepthInfo': return
	if self.c_type == 'SetupInfo': return
	if self.c_type == 'ScreenInfo': return
	
	# omit variable length struct writers, they're never used
	if not self.fixed_size():
		go('// omitting variable length send%s', self.c_type)
		go('')
		return
	
	go('func (c *Conn) send%sList(list []%s, count int) {', self.c_type, self.c_type)
	go('	b0 := make([]byte, %d * count)', structsize(self.fields))
	go('	for k := 0; k < count; k++ {')
	go('	b := b0[k * %d:]', structsize(self.fields))
	go_complex_writer_help(self.fields, 'list[k]')
	go('	}')
	go('	c.sendBytes(b0)')
	go('}')
	go('')

def go_union(self, name):
	pass

#
# Request writers with reply structs and readers where needed
#

def replyfields(self):
	l = []
	for field in self.fields:
		if field.type.is_pad or not field.wire: continue
		if field.field_name == 'response_type': continue
		if field.field_name == 'sequence': continue
		if field.field_name == 'length':
			if self.c_reply_name != 'GetImageReply' and self.c_reply_name != 'GetKeyboardMappingReply':
				continue
		l.append(field)
	return l

def go_reply(self, name):
	'''
	Declares the function that returns the reply structure.
	'''
	fields = replyfields(self.reply)
	go_complex(self.reply, fields)
	go('func (c *Conn) %s(cookie Cookie) (*%s, os.Error) {', self.c_reply_name, self.c_reply_type)
	go('	b, error := c.waitForReply(cookie)')
	go('	if error != nil { return nil, error }')
	go('	v := new(%s)', self.c_reply_type)
	go_complex_reader_help(self.reply, fields)
	go('	return v, nil')
	go('}')
	go('')

def go_request(self, name):
	'''
	Exported function that handles request declarations.
	'''
	go_type_setup(self, name, ('Request',))
	if symbols and n(name) not in symbols:
		go('// excluding request %s\n', n(name))
		return
	
	if self.reply:
		go_complex_writer(self, name, False)
		go_type_setup(self.reply, name, ('Reply',))
		go_reply(self, name)
	else:
		go_complex_writer(self, name, True)

#
# Event structs and readers
#

def eventfields(self):
	l = []
	for field in self.fields:
		if field.type.is_pad or not field.wire: continue
		if field.field_name == 'response_type': continue
		if field.field_name == 'sequence': continue
		l.append(field)
	return l

eventlist = []

def dumpeventlist():
	go('func parseEvent(buf []byte) (Event, os.Error) {')
	go('	switch buf[0] {')
	for event in eventlist:
		go('	case %s: return get%sEvent(buf), nil', event, event)
	go('	}')
	go('	return nil, os.NewError("unknown event type")')
	go('}')

def go_event(self, name):
	'''
	Exported function that handles event declarations.
	'''
	go_type_setup(self, name, ('Event',))
	if symbols and t(name) not in symbols:
		go('// excluding event %s\n', t(name))
		return
	
	eventlist.append(n(name))
	
	go('const %s = %s', t(name), self.opcodes[name])
	go('')
	fields = eventfields(self)
	if self.name == name:
		# Structure definition
		go_complex(self, fields)
		go('func get%s(b []byte) %s {', self.c_type, self.c_type)
		go('	var v %s', self.c_type)
		go_complex_reader_help(self, fields)
		go('	return v')
		go('}')
		go('')
	else:
		# maybe skip this depending on how it interacts with type switching on interfaces
		go('type %s %s', n(name + ('Event',)), n(self.name + ('Event',)))
		go('')
		go('func get%s(b []byte) %s {', self.c_type, self.c_type)
		go('	return (%s)(get%s(b))', n(name + ('Event',)), n(self.name + ('Event',)))
		go('}')
		go('')

#
# Map simple types to primitive types
#

def go_simple(self, name):
	'''
	Exported function that handles cardinal type declarations.
	These are types which are typedef'd to one of the CARDx's, char, float, etc.
	We stick them into the mangletab. Lop off xcb prefix.
	'''
	go_type_setup(self, name, ())
	if self.name != name:
		if _ns.is_ext:
			name = name[2]
		else:
			name = name[1]
		if name == "KEYSYM":
			mangletab[name] = "Keysym"
		elif name == "TIMESTAMP":
			mangletab[name] = "Timestamp"
		elif self.size == 4:
			mangletab[name] = "Id"
		else:
			mangletab[name] = t(self.name)

#
# Dump enums as consts, calculate implicit values instead
# of using iota.
#

def go_enum(self, name):
	if symbols and t(name) not in symbols:
		go('// excluding enum %s\n', t(name))
		return
	go('const (')
	iota = 0
	for (enam, eval) in self.values:
		if str(eval) == '':
			iota = iota + 1
			eval = iota
		else:
			iota = int(eval)
		if name[1] == 'Atom':
			s = name[1] + "".join([x.capitalize() for x in enam.split("_")])
		else:
			s = n(name + (enam,))
		go('	%s = %s', s, eval)
	go(')')
	go('')

errorlist = []

def dumperrorlist():
	go('var errorNames = map[byte]string{')
	for error in errorlist:
		go('	Bad%s: "%s",', error, error)
	go('}')
	go('')

def go_error(self, name):
	'''
	Exported function that handles error declarations.
	'''
	errorlist.append(n(name))
	go('const Bad%s = %s', n(name), self.opcodes[name])
	go('')

#
# Create the go file
#

def go_open(self):
	'''
	Exported function that handles module open.
	Opens the files and writes out the auto-generated code.
	'''
	global _ns
	_ns = self.namespace

	go('// This file was generated automatically from %s.', _ns.file)
	go('')
	go('package xgb')
	go('')
	go('import "os"')
	go('')
	
	if _ns.is_ext:
		go('const %s_MAJOR_VERSION = %s', _ns.ext_name.upper(), _ns.major_version)
		go('const %s_MINOR_VERSION = %s', _ns.ext_name.upper(), _ns.minor_version)
		go('')

def go_close(self):
	'''
	Exported function that handles module close.
	'''
	global outfile
	if len(eventlist) > 0:
		dumpeventlist()
	if len(errorlist) > 0:
		dumperrorlist()
	if not outfile:
		outfile = '%s.go' % _ns.header
	gofile = open(outfile, 'w')
	for line in golines:
		gofile.write(line)
		gofile.write('\n')
	gofile.close()

# Main routine starts here

# Must create an "output" dictionary before any xcbgen imports.
output = {'open'	: go_open,
		  'close'	: go_close,
		  'simple'	: go_simple,
		  'enum'	: go_enum,
		  'struct'	: go_struct,
		  'union'	: go_union,
		  'request' : go_request,
		  'event'	: go_event,
		  'error'	: go_error
		  }

# Boilerplate below this point

# Check for the argument that specifies path to the xcbgen python package.
try:
	opts, args = getopt.getopt(sys.argv[1:], 'p:s:o:')
except getopt.GetoptError, err:
	print str(err)
	print 'Usage: go_client.py [-p path] [-s symbol_list_file] [-o output.go] file.xml'
	sys.exit(1)

for (opt, arg) in opts:
	if opt == '-p':
		sys.path.append(arg)
	if opt == '-s':
		readsymbols(arg)
	if opt == '-o':
		outfile = arg

# Import the module class
try:
	from xcbgen.state import Module
except ImportError:
	print 'Failed to load the xcbgen Python package!'
	print 'Make sure that xcb/proto installed it on your Python path,'
	print 'or pass the path with -p.'
	print ''
	raise

module = Module(args[0], output)
module.register()
module.resolve()
module.generate()
