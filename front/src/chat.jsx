/* global React, Icon, CDLogo, CANNED_RESPONSES, GRAPH_DATA */
// 청담 — Chat view: messages, RAG pipeline, sources, composer

const { useState, useEffect, useRef, useCallback } = React;

// ─────────── Welcome ───────────
function Welcome({ onPick, logoVariant, brandPalette, suggestions }) {
  return (
    <div className="welcome">
      <div className="welcome-logo">
        <CDLogo size="lg" variant={logoVariant} color={brandPalette} />
      </div>
      <h1 className="welcome-title">무엇이 궁금하세요?</h1>
      <p className="welcome-sub">
        경북대학교 공지사항을 <b>청담</b>이 실시간으로 찾아드려요.<br/>
        일청담의 맑은 물처럼, 출처 없는 답은 흘려보냅니다.
      </p>
      <div className="suggestions">
        {suggestions.map(q => {
          const Ico = Icon[q.icon] || Icon.Search;
          return (
            <button key={q.id} className="suggestion" onClick={() => onPick(q)}>
              <span className="suggestion-icon"><Ico /></span>
              <span className="suggestion-body">
                <span className="suggestion-title">{q.title}</span>
                <span className="suggestion-hint">{q.hint}</span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─────────── RAG Pipeline (loading) ───────────
const RAG_STEPS = [
  { id: 'crawl',    label: '학교 홈페이지에서 최신 공지를 찾는 중',  tag: 'crawl' },
  { id: 'retrieve', label: '관련 문서를 검색해 컨텍스트를 모으는 중', tag: 'retrieve' },
  { id: 'rerank',   label: '문서 신뢰도를 비교하며 정렬하는 중',     tag: 'rerank' },
  { id: 'generate', label: '근거를 바탕으로 답변을 작성하는 중',     tag: 'generate' },
];

function RAGPipeline({ stepIdx }) {
  return (
    <div className="rag-pipeline">
      {RAG_STEPS.map((s, i) => {
        let state = 'pending';
        if (i < stepIdx) state = 'done';
        else if (i === stepIdx) state = 'active';
        return (
          <div key={s.id} className={`rag-step is-${state}`}>
            <span className="rag-step-icon">
              {state === 'done' ? <Icon.Check />
                : state === 'active' ? null
                : <span style={{width:6, height:6, borderRadius:'50%', background:'currentColor', display:'block'}} />}
            </span>
            <span className="rag-step-text">{s.label}</span>
            <span className="rag-step-tag">{s.tag}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─────────── Bubble content (markdown-lite) ───────────
function renderMarkdownLite(text) {
  // Very small renderer for **bold**, lists, line breaks
  const blocks = [];
  const lines = text.split('\n');
  let listBuf = null;
  let key = 0;

  const flushList = () => {
    if (listBuf && listBuf.length) {
      blocks.push(<ul key={'ul-' + (key++)}>
        {listBuf.map((li, i) => <li key={i}>{renderInline(li)}</li>)}
      </ul>);
    }
    listBuf = null;
  };

  for (const raw of lines) {
    const ln = raw;
    if (/^\s*-\s+/.test(ln)) {
      if (!listBuf) listBuf = [];
      listBuf.push(ln.replace(/^\s*-\s+/, ''));
    } else {
      flushList();
      if (ln.trim() === '') {
        blocks.push(<div key={'sp-' + (key++)} style={{height: 6}} />);
      } else {
        blocks.push(<div key={'l-' + (key++)}>{renderInline(ln)}</div>);
      }
    }
  }
  flushList();
  return blocks;
}

function renderInline(text) {
  // Bold **text**
  const parts = [];
  const re = /\*\*([^*]+)\*\*/g;
  let last = 0; let m;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(<strong key={k++}>{m[1]}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

// ─────────── Mini graph (next to answer) ───────────
function MiniGraph({ graphKey, onExpand }) {
  const data = GRAPH_DATA[graphKey] || GRAPH_DATA.general;
  if (!data) return null;
  const W = 360, H = 110, PAD = 10;
  const px = (x) => PAD + x * (W - 2 * PAD);
  const py = (y) => PAD + y * (H - 2 * PAD);

  return (
    <div className="mini-graph">
      <div className="mini-graph-head">
        <span style={{display:'flex', alignItems:'center', gap:6}}>
          <Icon.Graph />
          답변이 연결된 공지 지도
        </span>
        <button className="mini-graph-expand" onClick={onExpand}>전체 보기 →</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {data.edges.map(([a, b], i) => {
          const na = data.nodes.find(n => n.id === a);
          const nb = data.nodes.find(n => n.id === b);
          if (!na || !nb) return null;
          return <line key={i} x1={px(na.x)} y1={py(na.y)} x2={px(nb.x)} y2={py(nb.y)}
                       stroke="var(--cd-line)" strokeWidth="1" />;
        })}
        {data.nodes.map(n => {
          const r = Math.max(4, n.r * 0.28);
          const fill = n.type === 'topic' ? 'var(--cd-brand-500)'
                     : n.type === 'doc'   ? 'var(--cd-brand-200)'
                     : 'var(--cd-line)';
          const stroke = n.type === 'topic' ? 'var(--cd-brand-700)' : 'var(--cd-brand-300)';
          return <circle key={n.id} cx={px(n.x)} cy={py(n.y)} r={r} fill={fill} stroke={stroke} strokeWidth="1" />;
        })}
      </svg>
    </div>
  );
}

// ─────────── Sources block ───────────
function Sources({ sources, onOpenPreview }) {
  const [open, setOpen] = useState(true);
  if (!sources || sources.length === 0) return null;
  return (
    <div className={'sources' + (open ? ' is-open' : '')}>
      <div className="sources-header" onClick={() => setOpen(o => !o)}>
        <span className="sources-header-chev"><Icon.Chevron /></span>
        <Icon.Doc />
        참조한 공지 {sources.length}건
      </div>
      <div className="sources-list">
        {sources.map(s => (
          <div key={s.id} className="source-card" onClick={() => onOpenPreview(s)}>
            <div className="source-card-head">
              <div className="source-card-meta">
                <span className="source-badge" data-cat={s.category}>{s.category}</span>
                <span className="source-date">{s.publishedAt}</span>
              </div>
              <Icon.External />
            </div>
            <div className="source-title">{s.title}</div>
            <div className="source-summary">
              <HighlightedSummary text={s.summary} />
            </div>
            <div className="source-url">
              <Icon.Globe />
              {s.url.replace(/^https?:\/\//, '')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function HighlightedSummary({ text }) {
  // Render with **bold** as <mark>
  const parts = [];
  const re = /\*\*([^*]+)\*\*/g;
  let last = 0; let m; let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(<mark key={k++}>{m[1]}</mark>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  // Quote-wrap
  return <span>&ldquo;{parts}&rdquo;</span>;
}

// ─────────── Message bubble ───────────
function MessageBubble({ msg, isStreaming, ragStep, onBookmark, onCopy, onOpenPreview, onExpandGraph, showMiniGraph }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    onCopy && onCopy(msg);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  if (msg.role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="msg-body">
          <div className="msg-bubble">{msg.content}</div>
          <div className="msg-time">{msg.time}</div>
        </div>
        <div className="msg-avatar msg-avatar-user">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
          </svg>
        </div>
      </div>
    );
  }

  // assistant
  return (
    <div className="msg msg-assistant">
      <div className="msg-avatar" style={{position:'relative'}}>
        {isStreaming && <span className="bot-pulse" />}
        <Icon.BotDrop size={18} />
      </div>
      <div className="msg-body" style={{maxWidth:'90%'}}>
        {msg.isLoading ? (
          <RAGPipeline stepIdx={ragStep} />
        ) : (
          <>
            <div className="msg-bubble">
              {renderMarkdownLite(msg.content || '')}
              {msg.streaming && <span className="typing-cursor" />}
            </div>
            {!msg.streaming && (
              <div style={{display:'flex', alignItems:'center', gap:4}}>
                <span className="msg-time">{msg.time}</span>
                <div className="msg-actions">
                  <button className="msg-action" onClick={handleCopy} title="복사">
                    {copied ? <Icon.Check /> : <Icon.Copy />}
                  </button>
                  <button
                    className={'msg-action' + (msg.bookmarked ? ' is-active' : '')}
                    onClick={() => onBookmark(msg)}
                    title="즐겨찾기"
                  >
                    {msg.bookmarked ? <Icon.BookmarkFill /> : <Icon.Bookmark />}
                  </button>
                </div>
              </div>
            )}
            {!msg.streaming && msg.sources && msg.sources.length > 0 && (
              <Sources sources={msg.sources} onOpenPreview={onOpenPreview} />
            )}
            {!msg.streaming && showMiniGraph && msg.graphKey && (
              <MiniGraph graphKey={msg.graphKey} onExpand={() => onExpandGraph(msg.graphKey)} />
            )}
            {!msg.streaming && msg.followups && msg.followups.length > 0 && (
              <div className="followups">
                {msg.followups.map((f, i) => (
                  <button key={i} className="followup-chip" onClick={() => window.__sendFollowup && window.__sendFollowup(f)}>
                    <Icon.Sparkle style={{color:'var(--cd-brand-500)'}} />
                    {f}
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─────────── Composer ───────────
function Composer({ onSend, isStreaming, onAbort }) {
  const [value, setValue] = useState('');
  const taRef = useRef(null);

  const auto = useCallback(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 130) + 'px';
  }, []);

  useEffect(() => { auto(); }, [value, auto]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };
  const submit = () => {
    const v = value.trim();
    if (!v || isStreaming) return;
    onSend(v);
    setValue('');
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={taRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder="청담에게 학교 공지사항을 물어보세요..."
          rows={1}
          disabled={isStreaming}
        />
        {isStreaming ? (
          <button className="composer-send" onClick={onAbort} title="중단"
                  style={{background:'var(--cd-ink-2)'}}>
            <Icon.Stop />
          </button>
        ) : (
          <button className="composer-send" onClick={submit} disabled={!value.trim()} title="전송">
            <Icon.Send />
          </button>
        )}
      </div>
      <div className="composer-hint">
        <b>Shift+Enter</b>로 줄바꿈, <b>Enter</b>로 전송 · 청담은 출처가 있는 답변만 드려요 <span>·</span>
        <span>Hybrid RAG</span>
      </div>
    </div>
  );
}

Object.assign(window, {
  Welcome, MessageBubble, Composer, RAGPipeline, RAG_STEPS,
});
