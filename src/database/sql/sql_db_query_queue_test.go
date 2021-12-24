package sql

import (
	"context"
	"database/sql/driver"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

/**
Test simulate situation when duration between call query execution is less than execution time.
And analyzes max time waiting in queue.
*/

const (
	testDuration = 10 * time.Minute
	execDelay    = 500 * time.Millisecond
	appendDelay  = 400 * time.Millisecond
)

func TestDB_queryQueue(t *testing.T) {
	connector := &connectorMock{}
	db := OpenDB(connector)
	test := dbQueryQueueTest{
		t:     t,
		db:    db,
		index: 0,
		queue: make(map[uint64]queueMessage),
	}
	test.run()
}

type dbQueryQueueTest struct {
	t *testing.T

	db    *DB
	mx    sync.Mutex
	index uint64
	queue map[uint64]queueMessage
}

func (s *dbQueryQueueTest) run() {
	ctx, _ := context.WithTimeout(context.Background(), testDuration)

	s.db.SetMaxOpenConns(1)
	go s.execLoop(ctx)
	go s.monitor(ctx)

	<-ctx.Done()

	stat := s.getStats()
	if stat.maxDelay > 2*stat.expectedDelay {
		s.t.Error(fmt.Sprintf("current delay %v is more than 2 times greater than expected %v", stat.maxDelay, stat.expectedDelay))
	}
}

func (s *dbQueryQueueTest) execLoop(ctx context.Context) {
	for {
		now := time.Now()
		s.mx.Lock()
		s.index++
		msg := queueMessage{
			now,
			s.index,
		}
		s.queue[s.index] = msg
		s.mx.Unlock()
		go func() {
			_, _ = s.db.Exec("")
			s.mx.Lock()
			delete(s.queue, msg.index)
			s.mx.Unlock()
		}()
		select {
		default:
			time.Sleep(appendDelay)
		case <-ctx.Done():
			return
		}
	}
}

func (s *dbQueryQueueTest) monitor(ctx context.Context) {
	for {
		select {
		default:
			time.Sleep(10 * execDelay)
		case <-ctx.Done():
			return
		}
		stat := s.getStats()
		fmt.Printf(
			"delay(diff = %v, expected = %v, max = %v),  queue(len=%v, max=%v, min=%v, diff=%v)\n",
			stat.maxDelay-stat.expectedDelay,
			stat.expectedDelay,
			stat.maxDelay,
			stat.queueLen,
			stat.maxIndex,
			stat.minIndex,
			stat.maxIndex-stat.minIndex,
		)
	}
}

type stats struct {
	maxDelay      time.Duration
	expectedDelay time.Duration
	queueLen      int
	minIndex      uint64
	maxIndex      uint64
}

func (s *dbQueryQueueTest) getStats() (res stats) {
	s.mx.Lock()
	defer s.mx.Unlock()

	res.minIndex = math.MaxUint64

	now := time.Now()
	for index, event := range s.queue {
		delay := now.Sub(event.start)
		if delay > res.maxDelay {
			res.maxDelay = delay
		}
		if index > res.maxIndex {
			res.maxIndex = index
		}
		if index < res.minIndex {
			res.minIndex = index
		}
	}
	res.queueLen = len(s.queue)
	res.expectedDelay = time.Duration(res.queueLen-1) * appendDelay
	return
}

type resultMock struct {
}

func (r *resultMock) LastInsertId() (int64, error) {
	panic("implement resultMock.LastInsertId")
}

func (r *resultMock) RowsAffected() (int64, error) {
	panic("implement resultMock.RowsAffected")
}

type stmtMock struct {
}

func (s *stmtMock) Close() error {
	return nil
}

func (s *stmtMock) NumInput() int {
	return 0
}

func (s *stmtMock) Exec(_ []driver.Value) (driver.Result, error) {
	time.Sleep(execDelay)
	return &resultMock{}, nil
}

func (s *stmtMock) Query(_ []driver.Value) (driver.Rows, error) {
	panic("implement stmtMock.Query")
}

func newConnMock() driver.Conn {
	return &connMock{}
}

type connMock struct {
	lastExec time.Time
}

func (c *connMock) Prepare(_ string) (driver.Stmt, error) {
	return &stmtMock{}, nil
}

func (c *connMock) Close() error {
	return nil
}

func (c *connMock) Begin() (driver.Tx, error) {
	panic("implement connMockBegin")
}

type connectorMock struct {
}

func (c *connectorMock) Connect(_ context.Context) (driver.Conn, error) {
	return newConnMock(), nil
}

func (c *connectorMock) Driver() driver.Driver {
	panic("implement connectorMock.Driver")
}

type queueMessage struct {
	start time.Time
	index uint64
}
