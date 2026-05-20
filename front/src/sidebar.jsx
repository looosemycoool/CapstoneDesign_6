/* global React, Icon, CDLogo, HISTORY, BOOKMARKS */
// 청담 — Sidebar

const { useState, useMemo } = React;

function Sidebar({ collapsed, activeChat, onSelectChat, onNewChat, onOpenSettings, logoVariant, brandPalette }) {
  const [tab, setTab] = useState('chat'); // 'chat' | 'bookmarks'

  const grouped = useMemo(() => {
    const m = {};
    HISTORY.forEach(h => {
      if (!m[h.timeGroup]) m[h.timeGroup] = [];
      m[h.timeGroup].push(h);
    });
    return m;
  }, []);

  return (
    <aside className={'sidebar' + (collapsed ? ' is-collapsed' : '')}>
      <div className="sidebar-brand">
        <CDLogo size="sm" variant={logoVariant} color={brandPalette} />
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-name">청담<span style={{fontSize:11, color:'var(--cd-ink-4)', fontWeight:500, marginLeft:6, letterSpacing:'0.02em'}}>淸潭</span></span>
          <span className="sidebar-brand-sub">경북대 공지 어시스턴트</span>
        </div>
      </div>

      <div className="sidebar-actions">
        <button className="btn-new-chat" onClick={onNewChat}>
          <Icon.Plus />
          <span>새로운 채팅</span>
        </button>
      </div>

      <div className="sidebar-tabs">
        <button
          className={'sidebar-tab' + (tab === 'chat' ? ' is-active' : '')}
          onClick={() => setTab('chat')}
        >
          <Icon.Chat />
          대화 기록
        </button>
        <button
          className={'sidebar-tab' + (tab === 'bookmarks' ? ' is-active' : '')}
          onClick={() => setTab('bookmarks')}
        >
          <Icon.Bookmark />
          내 보관함
        </button>
      </div>

      {tab === 'chat' ? (
        <div className="sidebar-list" key="chat">
          {Object.entries(grouped).map(([group, items]) => (
            <React.Fragment key={group}>
              <div className="sidebar-section-label">{group}</div>
              {items.map(item => (
                <div
                  key={item.id}
                  className={'history-item' + (item.id === activeChat ? ' is-active' : '')}
                  onClick={() => onSelectChat(item.id)}
                >
                  <span className="history-item-icon"><Icon.Chat /></span>
                  <span className="history-item-text">{item.title}</span>
                  {item.id === activeChat
                    ? <button className="history-item-delete" onClick={(e) => { e.stopPropagation(); }}><Icon.Trash /></button>
                    : <span className="history-item-time">{item.timeLabel}</span>
                  }
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      ) : (
        <div className="sidebar-list" key="bookmarks">
          <div className="sidebar-section-label">저장된 답변</div>
          {BOOKMARKS.map(b => (
            <div key={b.id} className="history-item" style={{flexDirection:'column', alignItems:'flex-start', gap:4, padding:'12px'}}>
              <div style={{display:'flex', alignItems:'center', gap:6, width:'100%'}}>
                <span style={{
                  fontSize:10, fontWeight:600, padding:'2px 6px', borderRadius:4,
                  background:'var(--cd-brand-50)', color:'var(--cd-brand-700)',
                  letterSpacing:'0.02em'
                }}>{b.category}</span>
                <span style={{flex:1}} />
                <span style={{fontSize:10.5, color:'var(--cd-ink-4)'}}>{b.savedAt}</span>
              </div>
              <div style={{fontSize:13, fontWeight:600, color:'var(--cd-ink)', lineHeight:1.35}}>
                {b.title}
              </div>
              <div style={{fontSize:11.5, color:'var(--cd-ink-3)', lineHeight:1.5,
                           overflow:'hidden', textOverflow:'ellipsis',
                           display:'-webkit-box', WebkitLineClamp:2, WebkitBoxOrient:'vertical'}}>
                {b.snippet}
              </div>
            </div>
          ))}
          <div style={{padding:'24px 16px', textAlign:'center', color:'var(--cd-ink-4)', fontSize:11.5}}>
            답변 옆 <Icon.Bookmark style={{verticalAlign:'middle'}} /> 아이콘으로 저장
          </div>
        </div>
      )}

      <div className="sidebar-footer">
        <button className="sidebar-footer-btn" onClick={onOpenSettings}>
          <Icon.Settings />
          <span>설정</span>
        </button>
      </div>
    </aside>
  );
}

Object.assign(window, { Sidebar });
