/* global React, Icon, CDLogo */
// 청담 — Settings screen

function Settings({ onClose, tweak, setTweak, logoVariant, brandPalette }) {
  return (
    <div className="settings">
      <div style={{display:'flex', alignItems:'center', gap:12, marginBottom:24}}>
        <button className="icon-btn" onClick={onClose} style={{width:36, height:36}}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m15 18-6-6 6-6"/>
          </svg>
        </button>
        <h1 style={{margin:0}}>설정</h1>
      </div>

      <div className="settings-group">
        <div className="settings-group-label">계정</div>
        <div className="settings-row">
          <div style={{display:'flex', alignItems:'center', gap:14}}>
            <div style={{
              width:44, height:44, borderRadius:'50%',
              background: 'linear-gradient(135deg, #DBEAFB, #5BAEDC)',
              display:'flex', alignItems:'center', justifyContent:'center',
              color:'white', fontWeight:600, fontSize:16
            }}>경</div>
            <div className="settings-row-info">
              <span className="settings-row-title">경북대 재학생</span>
              <span className="settings-row-sub">student@knu.ac.kr · 컴퓨터학부 4학년</span>
            </div>
          </div>
          <button className="btn-pill">로그아웃</button>
        </div>
      </div>

      <div className="settings-group">
        <div className="settings-group-label">알림</div>
        <div className="settings-row">
          <div className="settings-row-info">
            <span className="settings-row-title">중요 공지 알림</span>
            <span className="settings-row-sub">즐겨찾기한 주제에 새 공지가 올라오면 알려드려요</span>
          </div>
          <button className="toggle is-on" />
        </div>
        <div className="settings-row">
          <div className="settings-row-info">
            <span className="settings-row-title">매일 학사 일정 요약</span>
            <span className="settings-row-sub">오전 8시에 그날의 마감일·일정 한 줄 요약</span>
          </div>
          <button className="toggle" />
        </div>
      </div>

      <div className="settings-group">
        <div className="settings-group-label">데이터 · 출처</div>
        <div className="settings-row">
          <div className="settings-row-info">
            <span className="settings-row-title">크롤링 도메인</span>
            <span className="settings-row-sub">knu.ac.kr · cs.knu.ac.kr · dorm.knu.ac.kr 외 4개</span>
          </div>
          <button className="btn-pill">관리</button>
        </div>
        <div className="settings-row">
          <div className="settings-row-info">
            <span className="settings-row-title">마지막 동기화</span>
            <span className="settings-row-sub">2026.05.19. 02:00 · 자동 (매일 새벽 2시)</span>
          </div>
          <button className="btn-pill"><Icon.Refresh /> 지금 동기화</button>
        </div>
        <div className="settings-row">
          <div className="settings-row-info">
            <span className="settings-row-title">출처 없는 답변 차단</span>
            <span className="settings-row-sub">근거가 부족하면 "모른다"라고 답해요 (권장)</span>
          </div>
          <button className="toggle is-on" />
        </div>
      </div>

      <div className="settings-group">
        <div className="settings-group-label">청담 정보</div>
        <div className="settings-row">
          <div style={{display:'flex', alignItems:'center', gap:12}}>
            <CDLogo size="sm" variant={logoVariant} color={brandPalette} animated={false} />
            <div className="settings-row-info">
              <span className="settings-row-title">청담 · v0.3.0 (beta)</span>
              <span className="settings-row-sub">
                "맑을 청 · 이야기 담" — 일청담 분수의 맑음처럼, 환각 없는 RAG
              </span>
            </div>
          </div>
        </div>
      </div>

      <div style={{textAlign:'center', padding:'24px 16px 8px', fontSize:11.5, color:'var(--cd-ink-4)'}}>
        Made by 6조 · 경북대학교 컴퓨터학부 종합프로젝트
      </div>
    </div>
  );
}

Object.assign(window, { Settings });
