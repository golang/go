#!/usr/bin/env python

# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This is a utility script for implementing a Go build slave.

import binascii
import httplib
import os
import struct
import subprocess
import sys
import time

buildhost = ''
buildport = -1
buildkey = ''

def main(args):
    global buildport, buildhost, buildkey

    if len(args) < 2:
        return usage(args[0])

    if 'BUILDHOST' not in os.environ:
        print >>sys.stderr, "Please set $BUILDHOST"
        return
    buildhost = os.environ['BUILDHOST']

    if 'BUILDPORT' not in os.environ:
        buildport = 80
    else:
        buildport = int(os.environ['BUILDPORT'])

    try:
        buildkey = file('%s/.gobuildkey' % os.environ['GOROOT'], 'r').read().strip()
    except IOError:
        try:
            buildkey = file('%s/.gobuildkey' % os.environ['HOME'], 'r').read().strip()
        except IOError:
            print >>sys.stderr, "Need key in $GOROOT/.gobuildkey or ~/.gobuildkey"
            return

    if args[1] == 'init':
        return doInit(args)
    elif args[1] == 'hwget':
        return doHWGet(args)
    elif args[1] == 'hwset':
        return doHWSet(args)
    elif args[1] == 'next':
        return doNext(args)
    elif args[1] == 'record':
        return doRecord(args)
    elif args[1] == 'benchmarks':
        return doBenchmarks(args)
    else:
        return usage(args[0])

def usage(name):
    sys.stderr.write('''Usage: %s <command>

Commands:
  init <rev>: init the build bot with the given commit as the first in history
  hwget <builder>: get the most recent revision built by the given builder
  hwset <builder> <rev>: get the most recent revision built by the given builder
  next <builder>: get the next revision number to by built by the given builder
  record <builder> <rev> <ok|log file>: record a build result
  benchmarks <builder> <rev> <log file>: record benchmark numbers
''' % name)
    return 1

def doInit(args):
    if len(args) != 3:
        return usage(args[0])
    c = getCommit(args[2])
    if c is None:
        fatal('Cannot get commit %s' % args[2])

    return command('init', {'node': c.node, 'date': c.date, 'user': c.user, 'desc': c.desc})

def doHWGet(args, retries = 0):
    if len(args) != 3:
        return usage(args[0])
    conn = httplib.HTTPConnection(buildhost, buildport, True)
    conn.request('GET', '/hw-get?builder=%s' % args[2]);
    reply = conn.getresponse()
    if reply.status == 200:
        print reply.read()
    elif reply.status == 500 and retries < 3:
        time.sleep(3)
        return doHWGet(args, retries = retries + 1)
    else:
        raise Failed('get-hw returned %d' % reply.status)
    return 0

def doHWSet(args):
    if len(args) != 4:
        return usage(args[0])
    c = getCommit(args[3])
    if c is None:
        fatal('Cannot get commit %s' % args[3])

    return command('hw-set', {'builder': args[2], 'hw': c.node})

def doNext(args):
    if len(args) != 3:
        return usage(args[0])
    conn = httplib.HTTPConnection(buildhost, buildport, True)
    conn.request('GET', '/hw-get?builder=%s' % args[2]);
    reply = conn.getresponse()
    if reply.status == 200:
        rev = reply.read()
    else:
        raise Failed('get-hw returned %d' % reply.status)

    c = getCommit(rev)
    next = getCommit(str(c.num + 1))
    if next is not None and next.parent == c.node:
        print c.num + 1
    else:
        print "<none>"
    return 0

def doRecord(args):
    if len(args) != 5:
        return usage(args[0])
    builder = args[2]
    rev = args[3]
    c = getCommit(rev)
    if c is None:
        print >>sys.stderr, "Bad revision:", rev
        return 1
    logfile = args[4]
    log = ''
    if logfile != 'ok':
        log = file(logfile, 'r').read()
    return command('build', {'node': c.node, 'parent': c.parent, 'date': c.date, 'user': c.user, 'desc': c.desc, 'log': log, 'builder': builder})

def doBenchmarks(args):
    if len(args) != 5:
        return usage(args[0])
    builder = args[2]
    rev = args[3]
    c = getCommit(rev)
    if c is None:
        print >>sys.stderr, "Bad revision:", rev
        return 1

    benchmarks = {}
    for line in file(args[4], 'r').readlines():
        if 'Benchmark' in line and 'ns/op' in line:
            parts = line.split()
            if parts[3] == 'ns/op':
                benchmarks[parts[0]] = (parts[1], parts[2])

    e = []
    for (name, (a, b)) in benchmarks.items():
        e.append(struct.pack('>H', len(name)))
        e.append(name)
        e.append(struct.pack('>H', len(a)))
        e.append(a)
        e.append(struct.pack('>H', len(b)))
        e.append(b)
    return command('benchmarks', {'node': c.node, 'builder': builder, 'benchmarkdata': binascii.b2a_base64(''.join(e))})

def encodeMultipartFormdata(fields, files):
    """fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files"""
    BOUNDARY = '----------ThIs_Is_tHe_bouNdaRY_$'
    CRLF = '\r\n'
    L = []
    for (key, value) in fields.items():
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"' % key)
        L.append('')
        L.append(value)
    for (key, filename, value) in files:
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
        L.append('Content-Type: %s' % get_content_type(filename))
        L.append('')
        L.append(value)
    L.append('--' + BOUNDARY + '--')
    L.append('')
    body = CRLF.join(L)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body

def unescapeXML(s):
    return s.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

class Commit:
    pass

def getCommit(rev):
    output, stderr = subprocess.Popen(['hg', 'log', '-r', rev, '-l', '1', '--template', '{rev}>{node|escape}>{author|escape}>{date}>{desc}'], stdout = subprocess.PIPE, stderr = subprocess.PIPE, close_fds = True).communicate()
    if len(stderr) > 0:
        return None
    [n, node, user, date, desc] = output.split('>', 4)

    c = Commit()
    c.num = int(n)
    c.node = unescapeXML(node)
    c.user = unescapeXML(user)
    c.date = unescapeXML(date)
    c.desc = desc
    c.parent = ''

    if c.num > 0:
        output, _ = subprocess.Popen(['hg', 'log', '-r', str(c.num - 1), '-l', '1', '--template', '{node}'], stdout = subprocess.PIPE, close_fds = True).communicate()
        c.parent = output

    return c

class Failed(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def command(cmd, args, retries = 0):
    args['key'] = buildkey
    contentType, body = encodeMultipartFormdata(args, [])
    print body
    conn = httplib.HTTPConnection(buildhost, buildport, True)
    conn.request('POST', '/' + cmd, body, {'Content-Type': contentType})
    reply = conn.getresponse()
    if reply.status != 200:
        print "Command failed. Output:"
        print reply.read()
    if reply.status == 500 and retries < 3:
        print "Was a 500. Waiting two seconds and trying again."
        time.sleep(2)
        return command(cmd, args, retries = retries + 1)
    if reply.status != 200:
        raise Failed('Command "%s" returned %d' % (cmd, reply.status))

if __name__ == '__main__':
    sys.exit(main(sys.argv))
