#!/usr/bin/env bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Illustrates how a Go language specification can be installed for Xcode 4.x.,
# to enable syntax coloring, by adding an entry to a plugindata file.
#
# FIXME: Write a decent Xcode plugin to handle the file type association and
# language specification properly instead of altering Xcode library files.

set -e

# Assumes Xcode 4+.
XCODE_MAJOR_VERSION=`xcodebuild -version | awk 'NR == 1 {print substr($2,1,1)}'`
if [ "$XCODE_MAJOR_VERSION" -lt "4" ]; then
	echo "Xcode 4.x not found."
	exit 1
fi

# DVTFOUNDATION_DIR may vary depending on Xcode setup. Change it to reflect
# your current Xcode setup. Find suitable path with e.g.:
#
#	find / -type f -name 'DVTFoundation.xcplugindata' 2> /dev/null
#
# Example of DVTFOUNDATION_DIR's from "default" Xcode 4.x setups;
#
#	Xcode 4.1: /Developer/Library/PrivateFrameworks/DVTFoundation.framework/Versions/A/Resources/
#	Xcode 4.3: /Applications/Xcode.app/Contents/SharedFrameworks/DVTFoundation.framework/Versions/A/Resources/
#
DVTFOUNDATION_DIR="/Applications/Xcode.app/Contents/SharedFrameworks/DVTFoundation.framework/Versions/A/Resources/"
PLUGINDATA_FILE="DVTFoundation.xcplugindata"

PLISTBUDDY=/usr/libexec/PlistBuddy
PLIST_FILE=tmp.plist

# Provide means of deleting the Go entry from the plugindata file.
if [ "$1" = "--delete-entry" ]; then
	echo "Removing Go language specification entry."
	$PLISTBUDDY -c "Delete :plug-in:extensions:Xcode.SourceCodeLanguage.Go" $DVTFOUNDATION_DIR/$PLUGINDATA_FILE
	echo "Run 'sudo rm -rf /var/folders/*' and restart Xcode to update change immediately."
	exit 0
fi

GO_VERSION="`go version`"

GO_LANG_ENTRY="
	<?xml version=\"1.0\" encoding=\"UTF-8\"?>
	<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
	<plist version=\"1.0\">
		<dict>
			<key>Xcode.SourceCodeLanguage.Go</key>
			<dict>
				<key>conformsTo</key>
				<array>
					<dict>
						<key>identifier</key>
						<string>Xcode.SourceCodeLanguage.Generic</string>
					</dict>
				</array>
				<key>documentationAbbreviation</key>
				<string>go</string>
				<key>fileDataType</key>
				<array>
					<dict>
						<key>identifier</key>
						<string>com.apple.xcode.go-source</string>
					</dict>
				</array>
				<key>id</key>
				<string>Xcode.SourceCodeLanguage.Go</string>
				<key>languageName</key>
				<string>Go</string>
				<key>languageSpecification</key>
				<string>xcode.lang.go</string>
				<key>name</key>
				<string>The Go Programming Language</string>
				<key>point</key>
				<string>Xcode.SourceCodeLanguage</string>
				<key>version</key>
				<string>$GO_VERSION</string>
			</dict>
		</dict>
	</plist>
"

echo "Backing up plugindata file."
cp $DVTFOUNDATION_DIR/$PLUGINDATA_FILE $DVTFOUNDATION_DIR/$PLUGINDATA_FILE.bak

echo "Adding Go language specification entry."
echo $GO_LANG_ENTRY > $PLIST_FILE
$PLISTBUDDY -c "Merge $PLIST_FILE plug-in:extensions" $DVTFOUNDATION_DIR/$PLUGINDATA_FILE

rm -f $PLIST_FILE

echo "Installing Go language specification file for Xcode."
cp $GOROOT/misc/xcode/4/go.xclangspec $DVTFOUNDATION_DIR

echo "Run 'sudo rm -rf /var/folders/*' and restart Xcode to update change immediately."
echo "Syntax coloring must be manually selected from the Editor - Syntax Coloring menu in Xcode."
