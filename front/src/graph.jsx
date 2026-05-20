/* global React, Icon, GRAPH_DATA */
// 청담 — Knowledge graph modal + Source preview panel

const { useState: useStateG, useEffect: useEffectG, useRef: useRefG } = React;

// ─────────── Knowledge Graph (full modal) ───────────
function KnowledgeGraph({ graphKey, onClose }) {
  const data = GRAPH_DATA[graphKey] || GRAPH_DATA.general;
  const [hovered, setHovered] = useStateG(null);
  const canvasRef = useRefG(null);
  const [size, setSize] = useStateG({ w: 800, h: 600 });

  useEffectG(() => {
    if (!canvasRef.current) return;
    const update = () => {
      const r = canvasRef.current.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(canvasRef.current);
    return () => ro.disconnect();
  }, []);

  const px = (x) => 60 + x * (size.w - 120);
  const py = (y) => 60 + y * (size.h - 120);

  const onBackdrop = (e) => {
    if (e.target === e.currentTarget) onClose();
  };

  useEffectG(() => {
    const k = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [onClose]);

  return (
    <div className="graph-modal-backdrop" onClick={onBackdrop}>
      <div className="graph-modal">
        <div className="graph-modal-head">
          <div className="graph-modal-title">
            <b>{data.title}</b>
            <span>{data.sub} · 답변의 근거가 된 공지와 개념의 관계망</span>
          </div>
          <button className="icon-btn" onClick={onClose}><Icon.X /></button>
        </div>
        <div className="graph-modal-canvas" ref={canvasRef}>
          <svg width="100%" height="100%" viewBox={`0 0 ${size.w} ${size.h}`}
               style={{display:'block', position:'absolute', inset:0}}>
            <defs>
              <radialGradient id="kgrad-topic">
                <stop offset="0%" stopColor="#5BAEDC"/>
                <stop offset="100%" stopColor="#2563EB"/>
              </radialGradient>
              <radialGradient id="kgrad-doc">
                <stop offset="0%" stopColor="#DBEAFB"/>
                <stop offset="100%" stopColor="#7FB7EE"/>
              </radialGradient>
              <radialGradient id="kgrad-concept">
                <stop offset="0%" stopColor="#F8FBFE"/>
                <stop offset="100%" stopColor="#E3EAF4"/>
              </radialGradient>
              <filter id="kshadow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Edges */}
            {data.edges.map(([a, b], i) => {
              const na = data.nodes.find(n => n.id === a);
              const nb = data.nodes.find(n => n.id === b);
              if (!na || !nb) return null;
              const isLit = hovered && (hovered === a || hovered === b);
              return (
                <line key={i}
                      x1={px(na.x)} y1={py(na.y)}
                      x2={px(nb.x)} y2={py(nb.y)}
                      stroke={isLit ? 'var(--cd-brand-400)' : 'var(--cd-line)'}
                      strokeWidth={isLit ? 2 : 1.2}
                      strokeDasharray={na.type === 'concept' || nb.type === 'concept' ? '4 4' : null}
                      opacity={hovered && !isLit ? 0.3 : 1}
                      style={{transition: 'all 200ms'}} />
              );
            })}

            {/* Nodes */}
            {data.nodes.map(n => {
              const r = n.r;
              const fill = n.type === 'topic' ? 'url(#kgrad-topic)'
                         : n.type === 'doc'   ? 'url(#kgrad-doc)'
                         :                       'url(#kgrad-concept)';
              const dim  = hovered && hovered !== n.id;
              return (
                <g key={n.id}
                   onMouseEnter={() => setHovered(n.id)}
                   onMouseLeave={() => setHovered(null)}
                   style={{cursor:'pointer', opacity: dim ? 0.4 : 1, transition: 'opacity 200ms'}}>
                  {n.type === 'topic' && (
                    <circle cx={px(n.x)} cy={py(n.y)} r={r + 14}
                            fill="none" stroke="var(--cd-brand-300)" strokeWidth="1"
                            opacity={hovered === n.id ? 0.6 : 0.25}>
                      <animate attributeName="r" from={r + 6} to={r + 18}
                               dur="2.4s" repeatCount="indefinite"/>
                      <animate attributeName="opacity" from="0.45" to="0"
                               dur="2.4s" repeatCount="indefinite"/>
                    </circle>
                  )}
                  <circle cx={px(n.x)} cy={py(n.y)} r={r}
                          fill={fill}
                          stroke={n.type === 'topic' ? '#1E40AF' : n.type === 'doc' ? '#4F95E3' : '#B4D5F6'}
                          strokeWidth="1.5"
                          filter={n.type === 'topic' ? 'url(#kshadow)' : null} />
                  {n.type === 'topic' && (
                    <text x={px(n.x)} y={py(n.y) + 4} fontSize="13" fontWeight="700"
                          textAnchor="middle" fill="white"
                          fontFamily="var(--cd-font)">{n.label}</text>
                  )}
                  {n.type !== 'topic' && (
                    <text x={px(n.x)} y={py(n.y) + r + 16} fontSize="11.5"
                          fontWeight={n.type === 'doc' ? 600 : 500}
                          textAnchor="middle"
                          fill={n.type === 'doc' ? 'var(--cd-ink)' : 'var(--cd-ink-3)'}
                          fontFamily="var(--cd-font)">{n.label}</text>
                  )}
                </g>
              );
            })}
          </svg>

          <div className="graph-modal-legend">
            <div className="graph-modal-legend-item">
              <span className="graph-modal-legend-dot" style={{background: 'var(--cd-brand-500)'}} />
              주제 (Topic)
            </div>
            <div className="graph-modal-legend-item">
              <span className="graph-modal-legend-dot" style={{background: 'var(--cd-brand-200)'}} />
              공지 문서 (Document)
            </div>
            <div className="graph-modal-legend-item">
              <span className="graph-modal-legend-dot" style={{background: 'var(--cd-line)', border: '1px solid var(--cd-brand-200)'}} />
              개념 (Concept)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────── Source preview side panel ───────────
function SourcePreview({ source, open, onClose }) {
  // Highlight all words in source.highlights
  const renderHighlighted = (body) => {
    if (!source || !source.highlights) return body;
    const sorted = [...source.highlights].sort((a, b) => b.length - a.length);
    // Build regex
    const pat = sorted.map(s => s.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')).join('|');
    if (!pat) return body;
    const re = new RegExp('(' + pat + ')', 'g');
    const parts = body.split(re);
    return parts.map((p, i) => re.test(p) ? <mark key={i}>{p}</mark> : <React.Fragment key={i}>{p}</React.Fragment>);
  };

  return (
    <div className={'preview-panel' + (open ? ' is-open' : '')}>
      <div className="preview-head">
        <div style={{display:'flex', alignItems:'center', gap:8, minWidth:0}}>
          {source && <span className="source-badge" data-cat={source.category}>{source?.category}</span>}
          <span style={{fontSize:11.5, color:'var(--cd-ink-3)'}}>출처 미리보기 · 인용 위치 강조</span>
        </div>
        <button className="icon-btn" onClick={onClose}><Icon.X /></button>
      </div>
      {source && (
        <div className="preview-body">
          <h3>{source.title}</h3>
          <div className="preview-meta">
            <span style={{display:'flex', alignItems:'center', gap:4}}>
              <Icon.Calendar style={{color:'var(--cd-ink-4)'}} /> {source.publishedAt}
            </span>
            <span style={{color:'var(--cd-ink-4)'}}>·</span>
            <a href={source.url} target="_blank" rel="noopener"
               style={{color:'var(--cd-brand-500)', fontSize:12, textDecoration:'none',
                       display:'flex', alignItems:'center', gap:4}}>
              <Icon.External /> 원문 보기
            </a>
          </div>
          <div className="preview-content">{renderHighlighted(source.body)}</div>
          <div style={{marginTop:24, padding:14, background:'var(--cd-bg-2)', borderRadius:12,
                       fontSize:12, color:'var(--cd-ink-3)', lineHeight:1.55, border:'1px dashed var(--cd-line)'}}>
            <div style={{display:'flex', gap:6, alignItems:'center', marginBottom:6,
                         fontWeight:600, color:'var(--cd-ink-2)'}}>
              <Icon.Sparkle style={{color:'var(--cd-brand-500)'}} />
              청담의 노트
            </div>
            답변에 인용된 구절을 형광펜으로 표시했어요. 출처 원문에서 직접 발췌한 부분만 강조됩니다 — 환각 없이.
          </div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, { KnowledgeGraph, SourcePreview });
