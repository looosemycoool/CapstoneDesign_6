/* global React, ReactDOM,
   Sidebar, Welcome, MessageBubble, Composer,
   KnowledgeGraph, SourcePreview, Settings,
   CDLogo, Icon,
   SUGGESTED_QUESTIONS, CANNED_RESPONSES, GRAPH_DATA,
   TweaksPanel, useTweaks, TweakSection, TweakColor, TweakRadio, TweakToggle */

// 청담 — Main app

const { useState, useEffect, useRef, useCallback } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "brandPalette": ["#5BAEDC", "#2563EB"],
  "logoVariant": "drop-ripple",
  "showSidebar": true,
  "showMiniGraph": true,
  "showLastUpdate": true,
  "darkMode": false,
  "density": "regular"
}/*EDITMODE-END*/;

const nowTime = () => {
  const d = new Date();
  const h = d.getHours();
  const m = d.getMinutes();
  const ampm = h < 12 ? '오전' : '오후';
  const h12 = ((h + 11) % 12) + 1;
  return `${ampm} ${h12}:${String(m).padStart(2,'0')}`;
};

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [ragStep, setRagStep] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeChat, setActiveChat] = useState('h-current');
  const [previewSource, setPreviewSource] = useState(null);
  const [graphKey, setGraphKey] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const convRef = useRef(null);
  const abortRef = useRef({ aborted: false });

  // Apply tweak: sidebar visibility
  useEffect(() => { setSidebarOpen(t.showSidebar); }, [t.showSidebar]);

  // Auto-scroll
  useEffect(() => {
    const el = convRef.current;
    if (!el) return;
    // Detect if user is scrolled up
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (isNearBottom) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Send a message (user) + simulate the RAG pipeline
  const sendMessage = useCallback(async (text, questionId) => {
    abortRef.current = { aborted: false };

    const userMsg = {
      id: 'u-' + Date.now(),
      role: 'user',
      content: text,
      time: nowTime(),
    };
    const loadingId = 'a-loading-' + Date.now();
    setMessages(prev => [...prev, userMsg, {
      id: loadingId, role: 'assistant', isLoading: true,
    }]);
    setIsStreaming(true);
    setRagStep(0);

    // Step through RAG pipeline
    const stepDelays = [700, 900, 700, 600];
    for (let i = 0; i < stepDelays.length; i++) {
      await new Promise(r => setTimeout(r, stepDelays[i]));
      if (abortRef.current.aborted) { setIsStreaming(false); return; }
      setRagStep(i + 1);
    }
    // Small pause
    await new Promise(r => setTimeout(r, 200));

    // Pick canned response
    const key = questionId || guessQuestionId(text);
    const canned = CANNED_RESPONSES[key] || CANNED_RESPONSES['__default__'];

    // Replace loading with streaming message
    const assistId = 'a-' + Date.now();
    setMessages(prev => prev.map(m => m.id === loadingId ? {
      id: assistId, role: 'assistant', content: '', streaming: true,
    } : m));

    // Stream tokens
    const full = canned.content;
    const chunks = chunkText(full);
    let acc = '';
    for (const ch of chunks) {
      if (abortRef.current.aborted) break;
      acc += ch;
      setMessages(prev => prev.map(m => m.id === assistId ? { ...m, content: acc } : m));
      await new Promise(r => setTimeout(r, 22 + Math.random() * 30));
    }

    // Finalize
    setMessages(prev => prev.map(m => m.id === assistId ? {
      ...m,
      content: acc,
      streaming: false,
      time: nowTime(),
      sources: canned.sources,
      followups: canned.followups,
      graphKey: canned.graph,
      bookmarked: false,
    } : m));
    setIsStreaming(false);
    setRagStep(0);
  }, []);

  // Expose followup sender for chip clicks
  useEffect(() => {
    window.__sendFollowup = (text) => {
      if (!isStreaming) sendMessage(text);
    };
  }, [sendMessage, isStreaming]);

  const handlePick = (q) => sendMessage(q.title, q.id);
  const handleAbort = () => { abortRef.current.aborted = true; setIsStreaming(false); };

  const handleNewChat = () => {
    setMessages([]);
    setActiveChat('h-current');
    setShowSettings(false);
  };

  const handleBookmark = (msg) => {
    setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, bookmarked: !m.bookmarked } : m));
  };

  const handleCopy = (msg) => {
    try { navigator.clipboard.writeText(msg.content || ''); } catch (e) {}
  };

  const isWelcome = messages.length === 0 && !showSettings;

  return (
    <div className={'app' + (t.darkMode ? ' is-dark' : '')}
         data-screen-label={isWelcome ? '01 Welcome' : showSettings ? '04 Settings' : '02 Chat'}>
      <Sidebar
        collapsed={!sidebarOpen}
        activeChat={activeChat}
        onSelectChat={(id) => { setActiveChat(id); setShowSettings(false); }}
        onNewChat={handleNewChat}
        onOpenSettings={() => setShowSettings(true)}
        logoVariant={t.logoVariant}
        brandPalette={t.brandPalette}
      />

      <main className="main">
        <div className="topbar">
          <div className="topbar-left">
            <button className="icon-btn" onClick={() => setSidebarOpen(o => !o)} title="사이드바">
              <Icon.Menu />
            </button>
            <CDLogo size="sm" variant={t.logoVariant} color={t.brandPalette} />
            <div className="topbar-title">
              <span className="topbar-title-main">청담 <span style={{color:'var(--cd-ink-4)', fontWeight:500, fontSize:12, marginLeft:4}}>淸潭</span></span>
              <span className="topbar-title-sub">경북대학교 공지사항 하이브리드 RAG 어시스턴트</span>
            </div>
          </div>
          <div className="topbar-right">
            {t.showLastUpdate && !showSettings && (
              <div className="last-update">
                <span className="last-update-dot" />
                마지막 동기화 2026.05.19. 02:00
              </div>
            )}
            {messages.length > 0 && !showSettings && (
              <button className="btn-pill" onClick={() => setGraphKey(messages.findLast?.(m => m.graphKey)?.graphKey || 'general')}>
                <Icon.Graph />
                지식 그래프
              </button>
            )}
            {!showSettings && (
              <button className="btn-pill"><Icon.Share /> 공유</button>
            )}
          </div>
        </div>

        {showSettings ? (
          <Settings
            onClose={() => setShowSettings(false)}
            tweak={t} setTweak={setTweak}
            logoVariant={t.logoVariant}
            brandPalette={t.brandPalette}
          />
        ) : (
          <>
            <div className="conversation" ref={convRef}>
              {isWelcome ? (
                <Welcome
                  onPick={handlePick}
                  logoVariant={t.logoVariant}
                  brandPalette={t.brandPalette}
                  suggestions={SUGGESTED_QUESTIONS}
                />
              ) : (
                <div className="messages">
                  {messages.map(m => (
                    <MessageBubble
                      key={m.id}
                      msg={m}
                      isStreaming={isStreaming && (m.isLoading || m.streaming)}
                      ragStep={ragStep}
                      onBookmark={handleBookmark}
                      onCopy={handleCopy}
                      onOpenPreview={(s) => setPreviewSource(s)}
                      onExpandGraph={(k) => setGraphKey(k)}
                      showMiniGraph={t.showMiniGraph}
                    />
                  ))}
                </div>
              )}
            </div>
            <Composer onSend={(v) => sendMessage(v)} isStreaming={isStreaming} onAbort={handleAbort} />
          </>
        )}
      </main>

      <SourcePreview
        source={previewSource}
        open={!!previewSource}
        onClose={() => setPreviewSource(null)}
      />
      {graphKey && (
        <KnowledgeGraph graphKey={graphKey} onClose={() => setGraphKey(null)} />
      )}

      {/* Tweaks panel */}
      <TweaksPanel title="Tweaks">
        <TweakSection label="Brand" />
        <TweakColor
          label="브랜드 컬러" value={t.brandPalette}
          options={[
            ['#5BAEDC', '#2563EB'],
            ['#7DD3FC', '#0284C7'],
            ['#A5B4FC', '#4F46E5'],
            ['#86EFAC', '#15803D'],
            ['#FBA1A1', '#DC2626'],
          ]}
          onChange={(v) => setTweak('brandPalette', v)}
        />
        <TweakRadio
          label="로고 스타일" value={t.logoVariant}
          options={[
            { value: 'drop-ripple', label: '물방울+파동' },
            { value: 'star-drop',   label: '별+물방울' },
            { value: 'cheong',      label: '청 글자' },
          ]}
          onChange={(v) => setTweak('logoVariant', v)}
        />

        <TweakSection label="Layout" />
        <TweakToggle label="사이드바 표시" value={t.showSidebar}
                     onChange={(v) => setTweak('showSidebar', v)} />
        <TweakToggle label="지식 그래프 mini-map" value={t.showMiniGraph}
                     onChange={(v) => setTweak('showMiniGraph', v)} />
        <TweakToggle label="마지막 동기화 표시" value={t.showLastUpdate}
                     onChange={(v) => setTweak('showLastUpdate', v)} />

        <TweakSection label="Theme" />
        <TweakToggle label="다크 모드" value={t.darkMode}
                     onChange={(v) => setTweak('darkMode', v)} />
      </TweaksPanel>
    </div>
  );
}

// ─────────── Helpers ───────────
function chunkText(text) {
  // Split into small chunks of 1-3 chars to simulate streaming
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    // Larger chunks on whitespace boundaries
    const isBreak = /[\s,.\n]/.test(text[i]);
    const n = isBreak ? 1 : Math.floor(1 + Math.random() * 3);
    chunks.push(text.slice(i, i + n));
    i += n;
  }
  return chunks;
}

function guessQuestionId(text) {
  const t = text.toLowerCase();
  if (/수강신청|시간표|폐강/.test(text)) return 'q-courses';
  if (/장학|국가장학금|장학금/.test(text)) return 'q-scholarship';
  if (/졸업|학점|toeic|졸업요건/i.test(text)) return 'q-graduation';
  if (/기숙사|생활관|입사|사생/.test(text)) return 'q-dorm';
  return '__default__';
}

// Mount
const root = ReactDOM.createRoot(document.getElementById('app'));
root.render(<App />);
