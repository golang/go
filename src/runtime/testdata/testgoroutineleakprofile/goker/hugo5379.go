// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"log"
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Hugo5379", Hugo5379)
}

type shortcodeHandler_hugo5379 struct {
	p                      *PageWithoutContent_hugo5379
	contentShortcodes      map[int]func() error
	contentShortcodesDelta map[int]func() error
	init                   sync.Once // O1
}

func (s *shortcodeHandler_hugo5379) executeShortcodesForDelta(p *PageWithoutContent_hugo5379) error {
	for k, _ := range s.contentShortcodesDelta {
		render := s.contentShortcodesDelta[k]
		if err := render(); err != nil {
			continue
		}
	}
	return nil
}

func (s *shortcodeHandler_hugo5379) updateDelta() {
	s.init.Do(func() {
		s.contentShortcodes = createShortcodeRenderers_hugo5379(s.p.withoutContent())
	})

	delta := make(map[int]func() error)

	for k, v := range s.contentShortcodes {
		if _, ok := delta[k]; !ok {
			delta[k] = v
		}
	}

	s.contentShortcodesDelta = delta
}

type Page_hugo5379 struct {
	*pageInit_hugo5379
	*pageContentInit_hugo5379
	pageWithoutContent *PageWithoutContent_hugo5379
	contentInit        sync.Once  // O2
	contentInitMu      sync.Mutex // L1
	shortcodeState     *shortcodeHandler_hugo5379
}

func (p *Page_hugo5379) WordCount() {
	p.initContentPlainAndMeta()
}

func (p *Page_hugo5379) initContentPlainAndMeta() {
	p.initContent()
	p.initPlain(true)
}

func (p *Page_hugo5379) initPlain(lock bool) {
	p.plainInit.Do(func() {
		if lock {
			/// Double locking here.
			p.contentInitMu.Lock()
			defer p.contentInitMu.Unlock()
		}
	})
}

func (p *Page_hugo5379) withoutContent() *PageWithoutContent_hugo5379 {
	p.pageInit_hugo5379.withoutContentInit.Do(func() {
		p.pageWithoutContent = &PageWithoutContent_hugo5379{Page_hugo5379: p}
	})
	return p.pageWithoutContent
}

func (p *Page_hugo5379) prepareForRender() error {
	var err error
	if err = handleShortcodes_hugo5379(p.withoutContent()); err != nil {
		return err
	}
	return nil
}

func (p *Page_hugo5379) setContentInit() {
	p.shortcodeState.updateDelta()
}

func (p *Page_hugo5379) initContent() {
	p.contentInit.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
		defer cancel()
		c := make(chan error, 1)

		go func() { // G2
			var err error
			p.contentInitMu.Lock() // first lock here
			defer p.contentInitMu.Unlock()

			err = p.prepareForRender()
			if err != nil {
				c <- err
				return
			}
			c <- err
		}()

		select {
		case <-ctx.Done():
		case <-c:
		}
	})
}

type PageWithoutContent_hugo5379 struct {
	*Page_hugo5379
}

type pageInit_hugo5379 struct {
	withoutContentInit sync.Once
}

type pageContentInit_hugo5379 struct {
	contentInit sync.Once // O3
	plainInit   sync.Once // O4
}

type HugoSites_hugo5379 struct {
	Sites []*Site_hugo5379
}

func (h *HugoSites_hugo5379) render() {
	for _, s := range h.Sites {
		for _, s2 := range h.Sites {
			s2.preparePagesForRender()
		}
		s.renderPages()
	}
}

func (h *HugoSites_hugo5379) Build() {
	h.render()
}

type Pages_hugo5379 []*Page_hugo5379

type PageCollections_hugo5379 struct {
	Pages Pages_hugo5379
}

type Site_hugo5379 struct {
	*PageCollections_hugo5379
}

func (s *Site_hugo5379) preparePagesForRender() {
	for _, p := range s.Pages {
		p.setContentInit()
	}
}

func (s *Site_hugo5379) renderForLayouts() {
	/// Omit reflections
	for _, p := range s.Pages {
		p.WordCount()
	}
}

func (s *Site_hugo5379) renderAndWritePage() {
	s.renderForLayouts()
}

func (s *Site_hugo5379) renderPages() {
	numWorkers := 2
	wg := &sync.WaitGroup{}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go pageRenderer_hugo5379(s, wg) // G3
	}

	wg.Wait()
}

type sitesBuilder_hugo5379 struct {
	H *HugoSites_hugo5379
}

func (s *sitesBuilder_hugo5379) Build() *sitesBuilder_hugo5379 {
	return s.build()
}

func (s *sitesBuilder_hugo5379) build() *sitesBuilder_hugo5379 {
	s.H.Build()
	return s
}

func (s *sitesBuilder_hugo5379) CreateSitesE() error {
	sites, err := NewHugoSites_hugo5379()
	if err != nil {
		return err
	}
	s.H = sites
	return nil
}

func (s *sitesBuilder_hugo5379) CreateSites() *sitesBuilder_hugo5379 {
	if err := s.CreateSitesE(); err != nil {
		log.Fatalf("Failed to create sites: %s", err)
	}
	return s
}

func newHugoSites_hugo5379(sites ...*Site_hugo5379) (*HugoSites_hugo5379, error) {
	h := &HugoSites_hugo5379{Sites: sites}
	return h, nil
}

func newSite_hugo5379() *Site_hugo5379 {
	c := &PageCollections_hugo5379{}
	s := &Site_hugo5379{
		PageCollections_hugo5379: c,
	}
	return s
}

func createSitesFromConfig_hugo5379() []*Site_hugo5379 {
	var (
		sites []*Site_hugo5379
	)

	var s *Site_hugo5379 = newSite_hugo5379()
	sites = append(sites, s)
	return sites
}

func NewHugoSites_hugo5379() (*HugoSites_hugo5379, error) {
	sites := createSitesFromConfig_hugo5379()
	return newHugoSites_hugo5379(sites...)
}

func prepareShortcodeForPage_hugo5379(p *PageWithoutContent_hugo5379) map[int]func() error {
	m := make(map[int]func() error)
	m[0] = func() error {
		return renderShortcode_hugo5379(p)
	}
	return m
}

func renderShortcode_hugo5379(p *PageWithoutContent_hugo5379) error {
	return renderShortcodeWithPage_hugo5379(p)
}

func renderShortcodeWithPage_hugo5379(p *PageWithoutContent_hugo5379) error {
	/// Omit reflections
	p.WordCount()
	return nil
}

func createShortcodeRenderers_hugo5379(p *PageWithoutContent_hugo5379) map[int]func() error {
	return prepareShortcodeForPage_hugo5379(p)
}

func newShortcodeHandler_hugo5379(p *Page_hugo5379) *shortcodeHandler_hugo5379 {
	return &shortcodeHandler_hugo5379{
		p:                      p.withoutContent(),
		contentShortcodes:      make(map[int]func() error),
		contentShortcodesDelta: make(map[int]func() error),
	}
}

func handleShortcodes_hugo5379(p *PageWithoutContent_hugo5379) error {
	return p.shortcodeState.executeShortcodesForDelta(p)
}

func pageRenderer_hugo5379(s *Site_hugo5379, wg *sync.WaitGroup) {
	defer wg.Done()
	s.renderAndWritePage()
}

func Hugo5379() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() { // G1
			b := &sitesBuilder_hugo5379{}
			s := b.CreateSites()
			for _, site := range s.H.Sites {
				p := &Page_hugo5379{
					pageInit_hugo5379:        &pageInit_hugo5379{},
					pageContentInit_hugo5379: &pageContentInit_hugo5379{},
					pageWithoutContent:       &PageWithoutContent_hugo5379{},
					contentInit:              sync.Once{},
					contentInitMu:            sync.Mutex{},
					shortcodeState:           nil,
				}
				p.shortcodeState = newShortcodeHandler_hugo5379(p)
				site.Pages = append(site.Pages, p)
			}
			s.Build()
		}()
	}
}
