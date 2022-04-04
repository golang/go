package sync

type Mutex struct{}

func (m *Mutex) Lock()
func (m *Mutex) Unlock()

type WaitGroup struct{}

func (wg *WaitGroup) Add(delta int)
func (wg *WaitGroup) Done()
func (wg *WaitGroup) Wait()
