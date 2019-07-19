package imports

import (
	"sync"
)

// ModuleResolver implements Resolver for modules using the go command as little
// as feasible.
//
// To find packages to import, the resolver needs to know about all of the
// the packages that could be imported. This includes packages that are
// already in modules that are in (1) the current module, (2) replace targets,
// and (3) packages in the module cache. Packages in (1) and (2) may change over
// time, as the client may edit the current module and locally replaced modules.
// The module cache (which includes all of the packages in (3)) can only
// ever be added to.
//
// The resolver can thus save state about packages in the module cache
// and guarantee that this will not change over time. To obtain information
// about new modules added to the module cache, the module cache should be
// rescanned.
//
// It is OK to serve information about modules that have been deleted,
// as they do still exist.
// TODO(suzmue): can we share information with the caller about
// what module needs to be downloaded to import this package?

type directoryPackageStatus int

const (
	_ directoryPackageStatus = iota
	directoryScanned
)

type directoryPackageInfo struct {
	// status indicates the extent to which this struct has been filled in.
	status directoryPackageStatus
	// err is non-nil when there was an error trying to reach status.
	err error

	// Set when status > directoryScanned.

	// dir is the absolute directory of this package.
	dir string
	// nonCanonicalImportPath is the expected import path for this package.
	// This may not be an import path that can be used to import this package.
	nonCanonicalImportPath string
	// needsReplace is true if the nonCanonicalImportPath does not match the
	// the modules declared path, making it impossible to import without a
	// replace directive.
	needsReplace bool
}

// reachedStatus returns true when info has a status at least target and any error associated with
// an attempt to reach target.
func (info *directoryPackageInfo) reachedStatus(target directoryPackageStatus) (bool, error) {
	if info.err == nil {
		return info.status >= target, nil
	}
	if info.status == target {
		return true, info.err
	}
	return true, nil
}

// moduleCacheInfo is a concurrency safe map for storing information about
// the directories in the module cache.
//
// The information in this cache is built incrementally. Entries are initialized in scan.
// No new keys should be added in any other functions, as all directories containing
// packages are identified in scan.
//
// Other functions, including loadExports and findPackage, may update entries in this cache
// as they discover new things about the directory.
//
// We do not need to protect the data in the cache for multiple writes, because it only stores
// module cache directories, which do not change. If two competing stores take place, there will be
// one store that wins. Although this could result in a loss of information it will not be incorrect
// and may just result in recomputing the same result later.
//
// TODO(suzmue): consider other concurrency strategies and data structures (RWLocks, sync.Map, etc)
type moduleCacheInfo struct {
	mu sync.Mutex
	// modCacheDirInfo stores information about packages in
	// module cache directories. Keyed by absolute directory.
	modCacheDirInfo map[string]*directoryPackageInfo
}

// Store stores the package info for dir.
func (d *moduleCacheInfo) Store(dir string, info directoryPackageInfo) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.modCacheDirInfo[dir] = &directoryPackageInfo{
		status:                 info.status,
		err:                    info.err,
		dir:                    info.dir,
		nonCanonicalImportPath: info.nonCanonicalImportPath,
		needsReplace:           info.needsReplace,
	}
}

// Load returns a copy of the directoryPackageInfo for absolute directory dir.
func (d *moduleCacheInfo) Load(dir string) (directoryPackageInfo, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	info, ok := d.modCacheDirInfo[dir]
	if !ok {
		return directoryPackageInfo{}, false
	}
	return *info, true
}

// Keys returns the keys currently present in d.
func (d *moduleCacheInfo) Keys() (keys []string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for key := range d.modCacheDirInfo {
		keys = append(keys, key)
	}
	return keys
}
