# Copyright 2025 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Mercurial extension to add a 'goreposum' command that
# computes a hash of a remote repo's tag state.
# Tag definitions can come from the .hgtags file stored in
# any head of any branch, and the server protocol does not
# expose the tags directly. However, the protocol does expose
# the hashes of all the branch heads, so we can use a hash of
# all those branch names and heads as a conservative snapshot
# of the entire remote repo state, and use that as the tag sum.
# Any change on the server then invalidates the tag sum,
# even if it didn't have anything to do with tags, but at least
# we will avoid re-cloning a server when there have been no
# changes at all.
#
# Separately, this extension also adds a 'golookup' command that
# returns the hash of a specific reference, like 'default' or a tag.
# And golookup of a hash confirms that it still exists on the server.
# We can use that to revalidate that specific versions still exist and
# have the same meaning they did the last time we checked.
#
# Usage:
#
#	hg --config "extensions.goreposum=$GOROOT/lib/hg/goreposum.py" goreposum REPOURL

import base64, hashlib, sys
from mercurial import registrar, ui, hg, node
from mercurial.i18n import _
cmdtable = {}
command = registrar.command(cmdtable)
@command(b'goreposum', [], _('url'), norepo=True)
def goreposum(ui, url):
  """
  goreposum computes a checksum of all the named state in the remote repo.
  It hashes together all the branch names and hashes
  and then all the bookmark names and hashes.
  Tags are stored in .hgtags files in any of the branches,
  so the branch metadata includes the tags as well.
  """
  h = hashlib.sha256()
  peer = hg.peer(ui, {}, url)
  for name, revs in peer.branchmap().items():
    h.update(name)
    for r in revs:
      h.update(b' ')
      h.update(r)
    h.update(b'\n')
  if (b'bookmarks' in peer.listkeys(b'namespaces')):
    for name, rev in peer.listkeys(b'bookmarks').items():
      h.update(name)
      h.update(b'=')
      h.update(rev)
      h.update(b'\n')
  print('r1:'+base64.standard_b64encode(h.digest()).decode('utf-8'))

@command(b'golookup', [], _('url rev'), norepo=True)
def golookup(ui, url, rev):
  """
  golookup looks up a single identifier in the repo,
  printing its hash.
  """
  print(node.hex(hg.peer(ui, {}, url).lookup(rev)).decode('utf-8'))
