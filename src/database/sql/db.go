package sql

import (
	"context"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"
)

// DB is a database handle representing a pool of zero or more
// underlying connections. It's safe for concurrent use by multiple
// goroutines.
//
// The sql package creates and frees connections automatically; it
// also maintains a free pool of idle connections. If the database has
// a concept of per-connection state, such state can be reliably observed
// within a transaction (Tx) or connection (Conn). Once DB.Begin is called, the
// returned Tx is bound to a single connection. Once Commit or
// Rollback is called on the transaction, that transaction's
// connection is returned to DB's idle connection pool. The pool size
// can be controlled with SetMaxIdleConns.
type DBStruct struct {
	// Total time waited for new connections.
	waitDuration atomic.Int64

	connector driver.Connector
	// numClosed is an atomic counter which represents a total number of
	// closed connections. Stmt.openStmt checks it before cleaning closed
	// connections in Stmt.css.
	numClosed atomic.Uint64

	mu           sync.Mutex    // protects following fields
	freeConn     []*driverConn // free connections ordered by returnedAt oldest to newest
	connRequests map[uint64]chan connRequest
	nextRequest  uint64 // Next key to use in connRequests.
	numOpen      int    // number of opened and pending open connections
	// Used to signal the need for new connections
	// a goroutine running connectionOpener() reads on this chan and
	// maybeOpenNewConnections sends on the chan (one send per needed connection)
	// It is closed during DBStruct.Close(). The close tells the connectionOpener
	// goroutine to exit.
	openerCh          chan struct{}
	closed            bool
	dep               map[finalCloser]depSet
	lastPut           map[*driverConn]string // stacktrace of last conn's put; debug only
	maxIdleCount      int                    // zero means defaultMaxIdleConns; negative means 0
	maxOpen           int                    // <= 0 means unlimited
	maxLifetime       time.Duration          // maximum amount of time a connection may be reused
	maxIdleTime       time.Duration          // maximum amount of time a connection may be idle before being closed
	cleanerCh         chan struct{}
	waitCount         int64 // Total number of connections waited for.
	maxIdleClosed     int64 // Total number of connections closed due to idle count.
	maxIdleTimeClosed int64 // Total number of connections closed due to idle time.
	maxLifetimeClosed int64 // Total number of connections closed due to max connection lifetime limit.

	stop func() // stop cancels the connection opener.
}

// addDep notes that x now depends on dep, and x's finalClose won't be
// called until all of x's dependencies are removed with removeDep.
func (db_ *DBStruct) addDep(x finalCloser, dep any) {
	db_.mu.Lock()
	defer db_.mu.Unlock()
	db_.addDepLocked(x, dep)
}

func (db_ *DBStruct) addDepLocked(x finalCloser, dep any) {
	if db_.dep == nil {
		db_.dep = make(map[finalCloser]depSet)
	}
	xdep := db_.dep[x]
	if xdep == nil {
		xdep = make(depSet)
		db_.dep[x] = xdep
	}
	xdep[dep] = true
}

// removeDep notes that x no longer depends on dep.
// If x still has dependencies, nil is returned.
// If x no longer has any dependencies, its finalClose method will be
// called and its error value will be returned.
func (db_ *DBStruct) removeDep(x finalCloser, dep any) error {
	db_.mu.Lock()
	fn := db_.removeDepLocked(x, dep)
	db_.mu.Unlock()
	return fn()
}

func (db_ *DBStruct) removeDepLocked(x finalCloser, dep any) func() error {
	xdep, ok := db_.dep[x]
	if !ok {
		panic(fmt.Sprintf("unpaired removeDep: no deps for %T", x))
	}

	l0 := len(xdep)
	delete(xdep, dep)

	switch len(xdep) {
	case l0:
		// Nothing removed. Shouldn't happen.
		panic(fmt.Sprintf("unpaired removeDep: no %T dep on %T", dep, x))
	case 0:
		// No more dependencies.
		delete(db_.dep, x)
		return x.finalClose
	default:
		// Dependencies remain.
		return func() error { return nil }
	}
}

func (db_ *DBStruct) pingDC(ctx context.Context, dc *driverConn, release func(error)) error {
	var err error
	if pinger, ok := dc.ci.(driver.Pinger); ok {
		withLock(dc, func() {
			err = pinger.Ping(ctx)
		})
	}
	release(err)
	return err
}

// PingContext verifies a connection to the database is still alive,
// establishing a connection if necessary.
func (db_ *DBStruct) PingContext(ctx context.Context) error {
	var dc *driverConn
	var err error

	err = db_.retry(func(strategy connReuseStrategy) error {
		dc, err = db_.conn(ctx, strategy)
		return err
	})

	if err != nil {
		return err
	}

	return db_.pingDC(ctx, dc, dc.releaseConn)
}

// Ping verifies a connection to the database is still alive,
// establishing a connection if necessary.
//
// Ping uses context.Background internally; to specify the context, use
// PingContext.
func (db_ *DBStruct) Ping() error {
	return db_.PingContext(context.Background())
}

// Close closes the database and prevents new queries from starting.
// Close then waits for all queries that have started processing on the server
// to finish.
//
// It is rare to Close a DBStruct, as the DBStruct handle is meant to be
// long-lived and shared between many goroutines.
func (db_ *DBStruct) Close() error {
	db_.mu.Lock()
	if db_.closed { // Make DBStruct.Close idempotent
		db_.mu.Unlock()
		return nil
	}
	if db_.cleanerCh != nil {
		close(db_.cleanerCh)
	}
	var err error
	fns := make([]func() error, 0, len(db_.freeConn))
	for _, dc := range db_.freeConn {
		fns = append(fns, dc.closeDBLocked())
	}
	db_.freeConn = nil
	db_.closed = true
	for _, req := range db_.connRequests {
		close(req)
	}
	db_.mu.Unlock()
	for _, fn := range fns {
		err1 := fn()
		if err1 != nil {
			err = err1
		}
	}
	db_.stop()
	if c, ok := db_.connector.(io.Closer); ok {
		err1 := c.Close()
		if err1 != nil {
			err = err1
		}
	}
	return err
}

const defaultMaxIdleConns = 2

func (db_ *DBStruct) maxIdleConnsLocked() int {
	n := db_.maxIdleCount
	switch {
	case n == 0:
		// TODO(bradfitz): ask driver, if supported, for its default preference
		return defaultMaxIdleConns
	case n < 0:
		return 0
	default:
		return n
	}
}

func (db_ *DBStruct) shortestIdleTimeLocked() time.Duration {
	if db_.maxIdleTime <= 0 {
		return db_.maxLifetime
	}
	if db_.maxLifetime <= 0 {
		return db_.maxIdleTime
	}

	min := db_.maxIdleTime
	if min > db_.maxLifetime {
		min = db_.maxLifetime
	}
	return min
}

// SetMaxIdleConns sets the maximum number of connections in the idle
// connection pool.
//
// If MaxOpenConns is greater than 0 but less than the new MaxIdleConns,
// then the new MaxIdleConns will be reduced to match the MaxOpenConns limit.
//
// If n <= 0, no idle connections are retained.
//
// The default max idle connections is currently 2. This may change in
// a future release.
func (db_ *DBStruct) SetMaxIdleConns(n int) {
	db_.mu.Lock()
	if n > 0 {
		db_.maxIdleCount = n
	} else {
		// No idle connections.
		db_.maxIdleCount = -1
	}
	// Make sure maxIdle doesn't exceed maxOpen
	if db_.maxOpen > 0 && db_.maxIdleConnsLocked() > db_.maxOpen {
		db_.maxIdleCount = db_.maxOpen
	}
	var closing []*driverConn
	idleCount := len(db_.freeConn)
	maxIdle := db_.maxIdleConnsLocked()
	if idleCount > maxIdle {
		closing = db_.freeConn[maxIdle:]
		db_.freeConn = db_.freeConn[:maxIdle]
	}
	db_.maxIdleClosed += int64(len(closing))
	db_.mu.Unlock()
	for _, c := range closing {
		c.Close()
	}
}

// SetMaxOpenConns sets the maximum number of open connections to the database.
//
// If MaxIdleConns is greater than 0 and the new MaxOpenConns is less than
// MaxIdleConns, then MaxIdleConns will be reduced to match the new
// MaxOpenConns limit.
//
// If n <= 0, then there is no limit on the number of open connections.
// The default is 0 (unlimited).
func (db_ *DBStruct) SetMaxOpenConns(n int) {
	db_.mu.Lock()
	db_.maxOpen = n
	if n < 0 {
		db_.maxOpen = 0
	}
	syncMaxIdle := db_.maxOpen > 0 && db_.maxIdleConnsLocked() > db_.maxOpen
	db_.mu.Unlock()
	if syncMaxIdle {
		db_.SetMaxIdleConns(n)
	}
}

// SetConnMaxLifetime sets the maximum amount of time a connection may be reused.
//
// Expired connections may be closed lazily before reuse.
//
// If d <= 0, connections are not closed due to a connection's age.
func (db_ *DBStruct) SetConnMaxLifetime(d time.Duration) {
	if d < 0 {
		d = 0
	}
	db_.mu.Lock()
	// Wake cleaner up when lifetime is shortened.
	if d > 0 && d < db_.maxLifetime && db_.cleanerCh != nil {
		select {
		case db_.cleanerCh <- struct{}{}:
		default:
		}
	}
	db_.maxLifetime = d
	db_.startCleanerLocked()
	db_.mu.Unlock()
}

// SetConnMaxIdleTime sets the maximum amount of time a connection may be idle.
//
// Expired connections may be closed lazily before reuse.
//
// If d <= 0, connections are not closed due to a connection's idle time.
func (db_ *DBStruct) SetConnMaxIdleTime(d time.Duration) {
	if d < 0 {
		d = 0
	}
	db_.mu.Lock()
	defer db_.mu.Unlock()

	// Wake cleaner up when idle time is shortened.
	if d > 0 && d < db_.maxIdleTime && db_.cleanerCh != nil {
		select {
		case db_.cleanerCh <- struct{}{}:
		default:
		}
	}
	db_.maxIdleTime = d
	db_.startCleanerLocked()
}

// startCleanerLocked starts connectionCleaner if needed.
func (db_ *DBStruct) startCleanerLocked() {
	if (db_.maxLifetime > 0 || db_.maxIdleTime > 0) && db_.numOpen > 0 && db_.cleanerCh == nil {
		db_.cleanerCh = make(chan struct{}, 1)
		go db_.connectionCleaner(db_.shortestIdleTimeLocked())
	}
}

func (db_ *DBStruct) connectionCleaner(d time.Duration) {
	const minInterval = time.Second

	if d < minInterval {
		d = minInterval
	}
	t := time.NewTimer(d)

	for {
		select {
		case <-t.C:
		case <-db_.cleanerCh: // maxLifetime was changed or DBStruct was closed.
		}

		db_.mu.Lock()

		d = db_.shortestIdleTimeLocked()
		if db_.closed || db_.numOpen == 0 || d <= 0 {
			db_.cleanerCh = nil
			db_.mu.Unlock()
			return
		}

		d, closing := db_.connectionCleanerRunLocked(d)
		db_.mu.Unlock()
		for _, c := range closing {
			c.Close()
		}

		if d < minInterval {
			d = minInterval
		}

		if !t.Stop() {
			select {
			case <-t.C:
			default:
			}
		}
		t.Reset(d)
	}
}

// connectionCleanerRunLocked removes connections that should be closed from
// freeConn and returns them along side an updated duration to the next check
// if a quicker check is required to ensure connections are checked appropriately.
func (db_ *DBStruct) connectionCleanerRunLocked(d time.Duration) (time.Duration, []*driverConn) {
	var idleClosing int64
	var closing []*driverConn
	if db_.maxIdleTime > 0 {
		// As freeConn is ordered by returnedAt process
		// in reverse order to minimise the work needed.
		idleSince := nowFunc().Add(-db_.maxIdleTime)
		last := len(db_.freeConn) - 1
		for i := last; i >= 0; i-- {
			c := db_.freeConn[i]
			if c.returnedAt.Before(idleSince) {
				i++
				closing = db_.freeConn[:i:i]
				db_.freeConn = db_.freeConn[i:]
				idleClosing = int64(len(closing))
				db_.maxIdleTimeClosed += idleClosing
				break
			}
		}

		if len(db_.freeConn) > 0 {
			c := db_.freeConn[0]
			if d2 := c.returnedAt.Sub(idleSince); d2 < d {
				// Ensure idle connections are cleaned up as soon as
				// possible.
				d = d2
			}
		}
	}

	if db_.maxLifetime > 0 {
		expiredSince := nowFunc().Add(-db_.maxLifetime)
		for i := 0; i < len(db_.freeConn); i++ {
			c := db_.freeConn[i]
			if c.createdAt.Before(expiredSince) {
				closing = append(closing, c)

				last := len(db_.freeConn) - 1
				// Use slow delete as order is required to ensure
				// connections are reused least idle time first.
				copy(db_.freeConn[i:], db_.freeConn[i+1:])
				db_.freeConn[last] = nil
				db_.freeConn = db_.freeConn[:last]
				i--
			} else if d2 := c.createdAt.Sub(expiredSince); d2 < d {
				// Prevent connections sitting the freeConn when they
				// have expired by updating our next deadline d.
				d = d2
			}
		}
		db_.maxLifetimeClosed += int64(len(closing)) - idleClosing
	}

	return d, closing
}

// DBStats contains database statistics.
type DBStats struct {
	MaxOpenConnections int // Maximum number of open connections to the database.

	// Pool Status
	OpenConnections int // The number of established connections both in use and idle.
	InUse           int // The number of connections currently in use.
	Idle            int // The number of idle connections.

	// Counters
	WaitCount         int64         // The total number of connections waited for.
	WaitDuration      time.Duration // The total time blocked waiting for a new connection.
	MaxIdleClosed     int64         // The total number of connections closed due to SetMaxIdleConns.
	MaxIdleTimeClosed int64         // The total number of connections closed due to SetConnMaxIdleTime.
	MaxLifetimeClosed int64         // The total number of connections closed due to SetConnMaxLifetime.
}

// Stats returns database statistics.
func (db_ *DBStruct) Stats() DBStats {
	wait := db_.waitDuration.Load()

	db_.mu.Lock()
	defer db_.mu.Unlock()

	stats := DBStats{
		MaxOpenConnections: db_.maxOpen,

		Idle:            len(db_.freeConn),
		OpenConnections: db_.numOpen,
		InUse:           db_.numOpen - len(db_.freeConn),

		WaitCount:         db_.waitCount,
		WaitDuration:      time.Duration(wait),
		MaxIdleClosed:     db_.maxIdleClosed,
		MaxIdleTimeClosed: db_.maxIdleTimeClosed,
		MaxLifetimeClosed: db_.maxLifetimeClosed,
	}
	return stats
}

// Assumes DBStruct.mu is locked.
// If there are connRequests and the connection limit hasn't been reached,
// then tell the connectionOpener to open new connections.
func (db_ *DBStruct) maybeOpenNewConnections() {
	numRequests := len(db_.connRequests)
	if db_.maxOpen > 0 {
		numCanOpen := db_.maxOpen - db_.numOpen
		if numRequests > numCanOpen {
			numRequests = numCanOpen
		}
	}
	for numRequests > 0 {
		db_.numOpen++ // optimistically
		numRequests--
		if db_.closed {
			return
		}
		db_.openerCh <- struct{}{}
	}
}

// Runs in a separate goroutine, opens new connections when requested.
func (db_ *DBStruct) connectionOpener(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-db_.openerCh:
			db_.openNewConnection(ctx)
		}
	}
}

// Open one new connection
func (db_ *DBStruct) openNewConnection(ctx context.Context) {
	// maybeOpenNewConnections has already executed DBStruct.numOpen++ before it sent
	// on DBStruct.openerCh. This function must execute DBStruct.numOpen-- if the
	// connection fails or is closed before returning.
	ci, err := db_.connector.Connect(ctx)
	db_.mu.Lock()
	defer db_.mu.Unlock()
	if db_.closed {
		if err == nil {
			ci.Close()
		}
		db_.numOpen--
		return
	}
	if err != nil {
		db_.numOpen--
		db_.putConnDBLocked(nil, err)
		db_.maybeOpenNewConnections()
		return
	}
	dc := &driverConn{
		db:         db_,
		createdAt:  nowFunc(),
		returnedAt: nowFunc(),
		ci:         ci,
	}
	if db_.putConnDBLocked(dc, err) {
		db_.addDepLocked(dc, dc)
	} else {
		db_.numOpen--
		ci.Close()
	}
}

// connRequest represents one request for a new connection
// When there are no idle connections available, DBStruct.conn will create
// a new connRequest and put it on the DBStruct.connRequests list.
type connRequest struct {
	conn *driverConn
	err  error
}

var errDBClosed = errors.New("sql: database is closed")

// nextRequestKeyLocked returns the next connection request key.
// It is assumed that nextRequest will not overflow.
func (db_ *DBStruct) nextRequestKeyLocked() uint64 {
	next := db_.nextRequest
	db_.nextRequest++
	return next
}

// conn returns a newly-opened or cached *driverConn.
func (db_ *DBStruct) conn(ctx context.Context, strategy connReuseStrategy) (*driverConn, error) {
	db_.mu.Lock()
	if db_.closed {
		db_.mu.Unlock()
		return nil, errDBClosed
	}
	// Check if the context is expired.
	select {
	default:
	case <-ctx.Done():
		db_.mu.Unlock()
		return nil, ctx.Err()
	}
	lifetime := db_.maxLifetime

	// Prefer a free connection, if possible.
	last := len(db_.freeConn) - 1
	if strategy == cachedOrNewConn && last >= 0 {
		// Reuse the lowest idle time connection so we can close
		// connections which remain idle as soon as possible.
		conn := db_.freeConn[last]
		db_.freeConn = db_.freeConn[:last]
		conn.inUse = true
		if conn.expired(lifetime) {
			db_.maxLifetimeClosed++
			db_.mu.Unlock()
			conn.Close()
			return nil, driver.ErrBadConn
		}
		db_.mu.Unlock()

		// Reset the session if required.
		if err := conn.resetSession(ctx); errors.Is(err, driver.ErrBadConn) {
			conn.Close()
			return nil, err
		}

		return conn, nil
	}

	// Out of free connections or we were asked not to use one. If we're not
	// allowed to open any more connections, make a request and wait.
	if db_.maxOpen > 0 && db_.numOpen >= db_.maxOpen {
		// Make the connRequest channel. It's buffered so that the
		// connectionOpener doesn't block while waiting for the req to be read.
		req := make(chan connRequest, 1)
		reqKey := db_.nextRequestKeyLocked()
		db_.connRequests[reqKey] = req
		db_.waitCount++
		db_.mu.Unlock()

		waitStart := nowFunc()

		// Timeout the connection request with the context.
		select {
		case <-ctx.Done():
			// Remove the connection request and ensure no value has been sent
			// on it after removing.
			db_.mu.Lock()
			delete(db_.connRequests, reqKey)
			db_.mu.Unlock()

			db_.waitDuration.Add(int64(time.Since(waitStart)))

			select {
			default:
			case ret, ok := <-req:
				if ok && ret.conn != nil {
					db_.putConn(ret.conn, ret.err, false)
				}
			}
			return nil, ctx.Err()
		case ret, ok := <-req:
			db_.waitDuration.Add(int64(time.Since(waitStart)))

			if !ok {
				return nil, errDBClosed
			}
			// Only check if the connection is expired if the strategy is cachedOrNewConns.
			// If we require a new connection, just re-use the connection without looking
			// at the expiry time. If it is expired, it will be checked when it is placed
			// back into the connection pool.
			// This prioritizes giving a valid connection to a client over the exact connection
			// lifetime, which could expire exactly after this point anyway.
			if strategy == cachedOrNewConn && ret.err == nil && ret.conn.expired(lifetime) {
				db_.mu.Lock()
				db_.maxLifetimeClosed++
				db_.mu.Unlock()
				ret.conn.Close()
				return nil, driver.ErrBadConn
			}
			if ret.conn == nil {
				return nil, ret.err
			}

			// Reset the session if required.
			if err := ret.conn.resetSession(ctx); errors.Is(err, driver.ErrBadConn) {
				ret.conn.Close()
				return nil, err
			}
			return ret.conn, ret.err
		}
	}

	db_.numOpen++ // optimistically
	db_.mu.Unlock()
	ci, err := db_.connector.Connect(ctx)
	if err != nil {
		db_.mu.Lock()
		db_.numOpen-- // correct for earlier optimism
		db_.maybeOpenNewConnections()
		db_.mu.Unlock()
		return nil, err
	}
	db_.mu.Lock()
	dc := &driverConn{
		db:         db_,
		createdAt:  nowFunc(),
		returnedAt: nowFunc(),
		ci:         ci,
		inUse:      true,
	}
	db_.addDepLocked(dc, dc)
	db_.mu.Unlock()
	return dc, nil
}

// putConnHook is a hook for testing.
var putConnHook func(*DBStruct, *driverConn)

// noteUnusedDriverStatement notes that ds is no longer used and should
// be closed whenever possible (when c is next not in use), unless c is
// already closed.
func (db_ *DBStruct) noteUnusedDriverStatement(c *driverConn, ds *driverStmt) {
	db_.mu.Lock()
	defer db_.mu.Unlock()
	if c.inUse {
		c.onPut = append(c.onPut, func() {
			ds.Close()
		})
	} else {
		c.Lock()
		fc := c.finalClosed
		c.Unlock()
		if !fc {
			ds.Close()
		}
	}
}

// debugGetPut determines whether getConn & putConn calls' stack traces
// are returned for more verbose crashes.
const debugGetPut = false

// putConn adds a connection to the DBStruct's free pool.
// err is optionally the last error that occurred on this connection.
func (db_ *DBStruct) putConn(dc *driverConn, err error, resetSession bool) {
	if !errors.Is(err, driver.ErrBadConn) {
		if !dc.validateConnection(resetSession) {
			err = driver.ErrBadConn
		}
	}
	db_.mu.Lock()
	if !dc.inUse {
		db_.mu.Unlock()
		if debugGetPut {
			fmt.Printf("putConn(%v) DUPLICATE was: %s\n\nPREVIOUS was: %s", dc, stack(), db_.lastPut[dc])
		}
		panic("sql: connection returned that was never out")
	}

	if !errors.Is(err, driver.ErrBadConn) && dc.expired(db_.maxLifetime) {
		db_.maxLifetimeClosed++
		err = driver.ErrBadConn
	}
	if debugGetPut {
		db_.lastPut[dc] = stack()
	}
	dc.inUse = false
	dc.returnedAt = nowFunc()

	for _, fn := range dc.onPut {
		fn()
	}
	dc.onPut = nil

	if errors.Is(err, driver.ErrBadConn) {
		// Don't reuse bad connections.
		// Since the conn is considered bad and is being discarded, treat it
		// as closed. Don't decrement the open count here, finalClose will
		// take care of that.
		db_.maybeOpenNewConnections()
		db_.mu.Unlock()
		dc.Close()
		return
	}
	if putConnHook != nil {
		putConnHook(db_, dc)
	}
	added := db_.putConnDBLocked(dc, nil)
	db_.mu.Unlock()

	if !added {
		dc.Close()
		return
	}
}

// Satisfy a connRequest or put the driverConn in the idle pool and return true
// or return false.
// putConnDBLocked will satisfy a connRequest if there is one, or it will
// return the *driverConn to the freeConn list if err == nil and the idle
// connection limit will not be exceeded.
// If err != nil, the value of dc is ignored.
// If err == nil, then dc must not equal nil.
// If a connRequest was fulfilled or the *driverConn was placed in the
// freeConn list, then true is returned, otherwise false is returned.
func (db_ *DBStruct) putConnDBLocked(dc *driverConn, err error) bool {
	if db_.closed {
		return false
	}
	if db_.maxOpen > 0 && db_.numOpen > db_.maxOpen {
		return false
	}
	if c := len(db_.connRequests); c > 0 {
		var req chan connRequest
		var reqKey uint64
		for reqKey, req = range db_.connRequests {
			break
		}
		delete(db_.connRequests, reqKey) // Remove from pending requests.
		if err == nil {
			dc.inUse = true
		}
		req <- connRequest{
			conn: dc,
			err:  err,
		}
		return true
	} else if err == nil && !db_.closed {
		if db_.maxIdleConnsLocked() > len(db_.freeConn) {
			db_.freeConn = append(db_.freeConn, dc)
			db_.startCleanerLocked()
			return true
		}
		db_.maxIdleClosed++
	}
	return false
}

// maxBadConnRetries is the number of maximum retries if the driver returns
// driver.ErrBadConn to signal a broken connection before forcing a new
// connection to be opened.
const maxBadConnRetries = 2

func (db_ *DBStruct) retry(fn func(strategy connReuseStrategy) error) error {
	for i := int64(0); i < maxBadConnRetries; i++ {
		err := fn(cachedOrNewConn)
		// retry if err is driver.ErrBadConn
		if err == nil || !errors.Is(err, driver.ErrBadConn) {
			return err
		}
	}

	return fn(alwaysNewConn)
}

// PrepareContext creates a prepared statement for later queries or executions.
// Multiple queries or executions may be run concurrently from the
// returned statement.
// The caller must call the statement's Close method
// when the statement is no longer needed.
//
// The provided context is used for the preparation of the statement, not for the
// execution of the statement.
func (db_ *DBStruct) PrepareContext(ctx context.Context, query string) (stmt Stmt, err error) {

	err = db_.retry(func(strategy connReuseStrategy) error {
		stmt, err = db_.prepare(ctx, query, strategy)
		return err
	})

	return
}

// Prepare creates a prepared statement for later queries or executions.
// Multiple queries or executions may be run concurrently from the
// returned statement.
// The caller must call the statement's Close method
// when the statement is no longer needed.
//
// Prepare uses context.Background internally; to specify the context, use
// PrepareContext.
func (db_ *DBStruct) Prepare(query string) (Stmt, error) {
	return db_.PrepareContext(context.Background(), query)
}

func (db_ *DBStruct) prepare(ctx context.Context, query string, strategy connReuseStrategy) (Stmt, error) {
	// TODO: check if DBStruct.driver supports an optional
	// driver.Preparer interface and call that instead, if so,
	// otherwise we make a prepared statement that's bound
	// to a connection, and to execute this prepared statement
	// we either need to use this connection (if it's free), else
	// get a new connection + re-prepare + execute on that one.
	dc, err := db_.conn(ctx, strategy)
	if err != nil {
		return nil, err
	}
	return db_.prepareDC(ctx, dc, dc.releaseConn, nil, query)
}

// prepareDC prepares a query on the driverConn and calls release before
// returning. When cg == nil it implies that a connection pool is used, and
// when cg != nil only a single driver connection is used.
func (db_ *DBStruct) prepareDC(ctx context.Context, dc *driverConn, release func(error), cg stmtConnGrabber, query string) (Stmt, error) {
	var ds *driverStmt
	var err error
	defer func() {
		release(err)
	}()
	withLock(dc, func() {
		ds, err = dc.prepareLocked(ctx, cg, query)
	})
	if err != nil {
		return nil, err
	}
	stmt_ := stmt{
		db:    db_,
		query: query,
		cg:    cg,
		cgds:  ds,
	}

	// When cg == nil this statement will need to keep track of various
	// connections they are prepared on and record the stmt_ dependency on
	// the DBStruct.
	if cg == nil {
		stmt_.css = []connStmt{{dc, ds}}
		stmt_.lastNumClosed = db_.numClosed.Load()
		db_.addDep(&stmt_, stmt_)
	}

	return &stmt_, nil
}

// ExecContext executes a query without returning any rows.
// The args are for any placeholder parameters in the query.
func (db_ *DBStruct) ExecContext(ctx context.Context, query string, args ...any) (Result, error) {
	var res Result
	var err error

	err = db_.retry(func(strategy connReuseStrategy) error {
		res, err = db_.exec(ctx, query, args, strategy)
		return err
	})

	return res, err
}

// Exec executes a query without returning any rows.
// The args are for any placeholder parameters in the query.
//
// Exec uses context.Background internally; to specify the context, use
// ExecContext.
func (db_ *DBStruct) Exec(query string, args ...any) (Result, error) {
	return db_.ExecContext(context.Background(), query, args...)
}

func (db_ *DBStruct) exec(ctx context.Context, query string, args []any, strategy connReuseStrategy) (Result, error) {
	dc, err := db_.conn(ctx, strategy)
	if err != nil {
		return nil, err
	}
	return db_.execDC(ctx, dc, dc.releaseConn, query, args)
}

func (db_ *DBStruct) execDC(ctx context.Context, dc *driverConn, release func(error), query string, args []any) (res Result, err error) {
	defer func() {
		release(err)
	}()
	execerCtx, ok := dc.ci.(driver.ExecerContext)
	var execer driver.Execer
	if !ok {
		execer, ok = dc.ci.(driver.Execer)
	}
	if ok {
		var nvdargs []driver.NamedValue
		var resi driver.Result
		withLock(dc, func() {
			nvdargs, err = driverArgsConnLocked(dc.ci, nil, args)
			if err != nil {
				return
			}
			resi, err = ctxDriverExec(ctx, execerCtx, execer, query, nvdargs)
		})
		if err != driver.ErrSkip {
			if err != nil {
				return nil, err
			}
			return driverResult{dc, resi}, nil
		}
	}

	var si driver.Stmt
	withLock(dc, func() {
		si, err = ctxDriverPrepare(ctx, dc.ci, query)
	})
	if err != nil {
		return nil, err
	}
	ds := &driverStmt{Locker: dc, si: si}
	defer ds.Close()
	return resultFromStatement(ctx, dc.ci, ds, args...)
}

// QueryContext executes a query that returns rows, typically a SELECT.
// The args are for any placeholder parameters in the query.
func (db_ *DBStruct) QueryContext(ctx context.Context, query string, args ...any) (*Rows, error) {
	var rows *Rows
	var err error

	err = db_.retry(func(strategy connReuseStrategy) error {
		rows, err = db_.query(ctx, query, args, strategy)
		return err
	})

	return rows, err
}

// Query executes a query that returns rows, typically a SELECT.
// The args are for any placeholder parameters in the query.
//
// Query uses context.Background internally; to specify the context, use
// QueryContext.
func (db_ *DBStruct) Query(query string, args ...any) (*Rows, error) {
	return db_.QueryContext(context.Background(), query, args...)
}

func (db_ *DBStruct) query(ctx context.Context, query string, args []any, strategy connReuseStrategy) (*Rows, error) {
	dc, err := db_.conn(ctx, strategy)
	if err != nil {
		return nil, err
	}

	return db_.queryDC(ctx, nil, dc, dc.releaseConn, query, args)
}

// queryDC executes a query on the given connection.
// The connection gets released by the releaseConn function.
// The ctx context is from a query method and the txctx context is from an
// optional transaction context.
func (db_ *DBStruct) queryDC(ctx, txctx context.Context, dc *driverConn, releaseConn func(error), query string, args []any) (*Rows, error) {
	queryerCtx, ok := dc.ci.(driver.QueryerContext)
	var queryer driver.Queryer
	if !ok {
		queryer, ok = dc.ci.(driver.Queryer)
	}
	if ok {
		var nvdargs []driver.NamedValue
		var rowsi driver.Rows
		var err error
		withLock(dc, func() {
			nvdargs, err = driverArgsConnLocked(dc.ci, nil, args)
			if err != nil {
				return
			}
			rowsi, err = ctxDriverQuery(ctx, queryerCtx, queryer, query, nvdargs)
		})
		if err != driver.ErrSkip {
			if err != nil {
				releaseConn(err)
				return nil, err
			}
			// Note: ownership of dc passes to the *Rows, to be freed
			// with releaseConn.
			rows := &Rows{
				dc:          dc,
				releaseConn: releaseConn,
				rowsi:       rowsi,
			}
			rows.initContextClose(ctx, txctx)
			return rows, nil
		}
	}

	var si driver.Stmt
	var err error
	withLock(dc, func() {
		si, err = ctxDriverPrepare(ctx, dc.ci, query)
	})
	if err != nil {
		releaseConn(err)
		return nil, err
	}

	ds := &driverStmt{Locker: dc, si: si}
	rowsi, err := rowsiFromStatement(ctx, dc.ci, ds, args...)
	if err != nil {
		ds.Close()
		releaseConn(err)
		return nil, err
	}

	// Note: ownership of ci passes to the *Rows, to be freed
	// with releaseConn.
	rows := &Rows{
		dc:          dc,
		releaseConn: releaseConn,
		rowsi:       rowsi,
		closeStmt:   ds,
	}
	rows.initContextClose(ctx, txctx)
	return rows, nil
}

// QueryRowContext executes a query that is expected to return at most one row.
// QueryRowContext always returns a non-nil value. Errors are deferred until
// Row's Scan method is called.
// If the query selects no rows, the *Row's Scan will return ErrNoRows.
// Otherwise, the *Row's Scan scans the first selected row and discards
// the rest.
func (db_ *DBStruct) QueryRowContext(ctx context.Context, query string, args ...any) *Row {
	rows, err := db_.QueryContext(ctx, query, args...)
	return &Row{rows: rows, err: err}
}

// QueryRow executes a query that is expected to return at most one row.
// QueryRow always returns a non-nil value. Errors are deferred until
// Row's Scan method is called.
// If the query selects no rows, the *Row's Scan will return ErrNoRows.
// Otherwise, the *Row's Scan scans the first selected row and discards
// the rest.
//
// QueryRow uses context.Background internally; to specify the context, use
// QueryRowContext.
func (db_ *DBStruct) QueryRow(query string, args ...any) *Row {
	return db_.QueryRowContext(context.Background(), query, args...)
}

// BeginTx starts a transaction.
//
// The provided context is used until the transaction is committed or rolled back.
// If the context is canceled, the sql package will roll back
// the transaction. Tx.Commit will return an error if the context provided to
// BeginTx is canceled.
//
// The provided TxOptions is optional and may be nil if defaults should be used.
// If a non-default isolation level is used that the driver doesn't support,
// an error will be returned.
func (db_ *DBStruct) BeginTx(ctx context.Context, opts *TxOptions) (*Tx, error) {
	var tx *Tx
	var err error

	err = db_.retry(func(strategy connReuseStrategy) error {
		tx, err = db_.begin(ctx, opts, strategy)
		return err
	})

	return tx, err
}

// Begin starts a transaction. The default isolation level is dependent on
// the driver.
//
// Begin uses context.Background internally; to specify the context, use
// BeginTx.
func (db_ *DBStruct) Begin() (*Tx, error) {
	return db_.BeginTx(context.Background(), nil)
}

func (db_ *DBStruct) begin(ctx context.Context, opts *TxOptions, strategy connReuseStrategy) (tx *Tx, err error) {
	dc, err := db_.conn(ctx, strategy)
	if err != nil {
		return nil, err
	}
	return db_.beginDC(ctx, dc, dc.releaseConn, opts)
}

// beginDC starts a transaction. The provided dc must be valid and ready to use.
func (db_ *DBStruct) beginDC(ctx context.Context, dc *driverConn, release func(error), opts *TxOptions) (tx *Tx, err error) {
	var txi driver.Tx
	keepConnOnRollback := false
	withLock(dc, func() {
		_, hasSessionResetter := dc.ci.(driver.SessionResetter)
		_, hasConnectionValidator := dc.ci.(driver.Validator)
		keepConnOnRollback = hasSessionResetter && hasConnectionValidator
		txi, err = ctxDriverBegin(ctx, opts, dc.ci)
	})
	if err != nil {
		release(err)
		return nil, err
	}

	// Schedule the transaction to rollback when the context is canceled.
	// The cancel function in Tx will be called after done is set to true.
	ctx, cancel := context.WithCancel(ctx)
	tx = &Tx{
		db:                 db_,
		dc:                 dc,
		releaseConn:        release,
		txi:                txi,
		cancel:             cancel,
		keepConnOnRollback: keepConnOnRollback,
		ctx:                ctx,
	}
	go tx.awaitDone()
	return tx, nil
}

// Driver returns the database's underlying driver.
func (db_ *DBStruct) Driver() driver.Driver {
	return db_.connector.Driver()
}

// ErrConnDone is returned by any operation that is performed on a connection
// that has already been returned to the connection pool.
var ErrConnDone = errors.New("sql: connection is already closed")

// Conn returns a single connection by either opening a new connection
// or returning an existing connection from the connection pool. Conn will
// block until either a connection is returned or ctx is canceled.
// Queries run on the same Conn will be run in the same database session.
//
// Every Conn must be returned to the database pool after use by
// calling Conn.Close.
func (db_ *DBStruct) Conn(ctx context.Context) (*Conn, error) {
	var dc *driverConn
	var err error

	err = db_.retry(func(strategy connReuseStrategy) error {
		dc, err = db_.conn(ctx, strategy)
		return err
	})

	if err != nil {
		return nil, err
	}

	conn := &Conn{
		db: db_,
		dc: dc,
	}
	return conn, nil
}

// A NamedArg is a named argument. NamedArg values may be used as
// arguments to Query or Exec and bind to the corresponding named
// parameter in the SQL statement.
//
// For a more concise way to create NamedArg values, see
// the Named function.
type NamedArg struct {
	_NamedFieldsRequired struct{}

	// Name is the name of the parameter placeholder.
	//
	// If empty, the ordinal position in the argument list will be
	// used.
	//
	// Name must omit any symbol prefix.
	Name string

	// Value is the value of the parameter.
	// It may be assigned the same value types as the query
	// arguments.
	Value any
}

// Named provides a more concise way to create NamedArg values.
//
// Example usage:
//
//	DBStruct.ExecContext(ctx, `
//	    delete from Invoice
//	    where
//	        TimeCreated < @end
//	        and TimeCreated >= @start;`,
//	    sql.Named("start", startTime),
//	    sql.Named("end", endTime),
//	)
func Named(name string, value any) NamedArg {
	// This method exists because the go1compat promise
	// doesn't guarantee that structs don't grow more fields,
	// so unkeyed struct literals are a vet error. Thus, we don't
	// want to allow sql.NamedArg{name, value}.
	return NamedArg{Name: name, Value: value}
}
