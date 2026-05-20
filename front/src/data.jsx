/* 청담 — Mock data */
// Suggested questions, canned RAG responses, sources, history, knowledge graph

const SUGGESTED_QUESTIONS = [
  {
    id: 'q-courses',
    icon: 'Calendar',
    title: '2026학년도 1학기 수강신청 일정 알려줘',
    hint: '학사 · 일정',
  },
  {
    id: 'q-scholarship',
    icon: 'Coin',
    title: '국가장학금 2학기 신청 방법은?',
    hint: '장학 · 신청절차',
  },
  {
    id: 'q-graduation',
    icon: 'Cap',
    title: '컴퓨터학부 졸업 요건 알려줘',
    hint: '학사 · 졸업',
  },
  {
    id: 'q-dorm',
    icon: 'Home',
    title: '기숙사 신청은 어떻게 해?',
    hint: '생활 · 기숙사',
  },
];

// Canned responses keyed by question id
const CANNED_RESPONSES = {
  'q-courses': {
    content:
`2026학년도 1학기 수강신청 일정을 안내드립니다.

**수강신청 기간:** 2026년 2월 10일(화) ~ 2월 14일(토)
재학생은 학년별로 지정된 시간에 신청해야 합니다.

- **4학년**: 2월 10일(화) 09:00 ~
- **3학년**: 2월 11일(수) 09:00 ~
- **2학년**: 2월 12일(목) 09:00 ~
- **1학년**: 2월 13일(금) 09:00 ~

수강신청은 경북대 포털(KNU 통합정보시스템)에서 진행되며, 정해진 시간에만 접속이 가능합니다. 수강정정 기간은 개강 후 첫째 주(3.2~3.6)입니다.`,
    sources: [
      {
        id: 's-1',
        category: '학사',
        title: '2026학년도 1학기 수강신청 안내',
        summary: '수강신청 기간은 2026년 2월 10일부터 2월 14일까지이며, 재학생은 학년별로 지정된 시간에 신청해야 합니다. **4학년 2월 10일, 3학년 2월 11일, 2학년 2월 12일, 1학년 2월 13일**...',
        publishedAt: '2026-01-08',
        url: 'https://knu.ac.kr/notice/12345',
        highlights: ['2026년 2월 10일', '2월 14일', '학년별로 지정된 시간', '4학년 2월 10일'],
        body: `2026학년도 1학기 수강신청에 관한 사항을 아래와 같이 안내합니다.

1. 수강신청 기간
   가. 수강신청 기간은 2026년 2월 10일(화)부터 2월 14일(토)까지이며, 재학생은 학년별로 지정된 시간에 신청해야 합니다.
   나. 신청 시간: 매일 09:00부터 익일 02:00까지 가능
   다. 학년별 일정:
       - 4학년: 2월 10일(화) 09:00
       - 3학년: 2월 11일(수) 09:00
       - 2학년: 2월 12일(목) 09:00
       - 1학년: 2월 13일(금) 09:00
       - 전 학년: 2월 14일(토) 자유신청

2. 수강신청 시스템
   - 경북대학교 통합정보시스템(KNU Portal) 접속
   - 학사정보 > 수강신청 메뉴 이용
   - 동시 접속자가 많은 경우 대기열이 발생할 수 있음

3. 유의사항
   가. 수강신청 시간 외에는 접속이 제한됩니다.
   나. 수강정정 기간(3월 2일~3월 6일)을 활용하시기 바랍니다.
   다. 수강신청 사전 안내사항을 반드시 확인하시기 바랍니다.

문의: 학사지원과 (053-950-2024)`,
      },
      {
        id: 's-2',
        category: '학사',
        title: '2026-1학기 수강정정 및 폐강 안내',
        summary: '수강정정 기간은 개강 후 첫째 주(3월 2일~3월 6일)이며, 폐강은 신청 인원이 10명 미만인 강좌에 한해 적용됩니다...',
        publishedAt: '2026-01-12',
        url: 'https://knu.ac.kr/notice/12346',
        highlights: ['수강정정 기간', '3월 2일', '3월 6일'],
        body: `수강정정 및 폐강에 관한 안내사항입니다.

1. 수강정정 기간: 2026년 3월 2일(월) ~ 3월 6일(금)
2. 정정 가능 시간: 매일 09:00 ~ 17:00
3. 폐강 기준: 수강신청 인원 10명 미만 강좌`,
      },
    ],
    followups: [
      '수강신청 전 사전 안내사항이 뭐야?',
      '수강 정정 기간은 언제까지야?',
      '시간표 어디서 확인해?',
    ],
    graph: 'courses',
  },

  'q-scholarship': {
    content:
`국가장학금 2학기 신청 방법을 안내드립니다.

**신청 기간:** 2026년 5월 27일(화) ~ 6월 26일(목)
**신청 대상:** 경북대학교 재학생 (휴학생 제외)

**신청 절차**
1. 한국장학재단 홈페이지(www.kosaf.go.kr) 접속
2. 학자금 지원 > 국가장학금 신청
3. 가구원 동의 절차 진행 (소득·재산 조사용)
4. 서류 제출 (필요 시)

**중요 안내**
- 1차 신청을 놓치신 분들도 2차 신청 가능합니다.
- 가구원 정보제공 동의가 필수입니다.
- 신청 후 학생 정보(연락처·계좌) 입력 필수.`,
    sources: [
      {
        id: 's-3',
        category: '장학',
        title: '2026학년도 2학기 국가장학금 신청 안내',
        summary: '2026학년도 2학기 국가장학금 신청 기간은 **5월 27일부터 6월 26일까지**이며, 한국장학재단 홈페이지에서 신청 가능합니다...',
        publishedAt: '2026-05-15',
        url: 'https://knu.ac.kr/notice/12350',
        highlights: ['5월 27일', '6월 26일', '한국장학재단', '가구원 동의'],
        body: `2026학년도 2학기 국가장학금 신청 안내

1. 신청 기간: 2026년 5월 27일(화) ~ 6월 26일(목) 18:00까지
2. 신청 사이트: 한국장학재단 (www.kosaf.go.kr)
3. 신청 자격: 경북대학교 재학생 (1차 미신청자 포함)
4. 필수 절차:
   - 가구원 동의 (소득·재산 조사를 위함)
   - 학생 정보 입력 (연락처, 계좌번호 등)
5. 서류 제출: 한국장학재단 별도 요청 시
6. 결과 발표: 7월 중순 예정`,
      },
      {
        id: 's-4',
        category: '장학',
        title: '교내장학금 신청 안내 (가계곤란 장학금)',
        summary: '교내 가계곤란 장학금은 국가장학금과 별도로 신청 가능하며, 신청 기간은 **6월 1일부터 6월 14일까지**입니다...',
        publishedAt: '2026-05-18',
        url: 'https://knu.ac.kr/notice/12352',
        highlights: ['6월 1일', '6월 14일', '가계곤란 장학금'],
        body: `교내 가계곤란 장학금 신청 안내`,
      },
    ],
    followups: [
      '교내장학금도 알려줘',
      '서류 제출은 어디서 해?',
      '가구원 동의는 어떻게 하는 거야?',
    ],
    graph: 'scholarship',
  },

  'q-graduation': {
    content:
`경북대 **컴퓨터학부 졸업 요건**을 안내드립니다.

**1. 총 이수학점**
- 졸업 최소 학점: **130학점** 이상

**2. 전공 이수**
- 전공 기초: 18학점
- 전공 필수: 21학점
- 전공 선택: 33학점 이상
- **전공 총합 72학점 이상**

**3. 교양 이수**
- 기초교양(영어/글쓰기/SW): 12학점
- 핵심교양: 12학점 이상
- 일반교양: 6학점 이상

**4. 졸업 인증 요건**
- TOEIC 700점 이상 (또는 동등 어학시험)
- 코딩능력 인증 (PCCP 등) 또는 졸업작품 PASS

복학생은 입학년도 기준의 교육과정을 따르며, 학과 사무실에서 졸업사정 확인이 필요합니다.`,
    sources: [
      {
        id: 's-5',
        category: '학사',
        title: '2026학년도 컴퓨터학부 졸업요건 안내',
        summary: '컴퓨터학부 졸업을 위해서는 **총 130학점 이상**과 전공 72학점, **TOEIC 700점** 또는 동등 자격을 충족해야 합니다...',
        publishedAt: '2026-03-04',
        url: 'https://knu.ac.kr/notice/12289',
        highlights: ['130학점', '전공 72학점', 'TOEIC 700점', '코딩능력 인증'],
        body: `컴퓨터학부 졸업요건 안내 (2026학년도 기준)`,
      },
    ],
    followups: [
      '코딩능력 인증은 어떻게 해?',
      '졸업 사정은 언제 신청해?',
      '복학생도 같은 요건이야?',
    ],
    graph: 'graduation',
  },

  'q-dorm': {
    content:
`경북대 기숙사(생활관) 신청 방법을 안내드립니다.

**2026-2학기 입사 신청**
- 신청 기간: 2026년 6월 16일(월) ~ 6월 27일(금)
- 신청 사이트: 경북대 생활관 홈페이지
- 입사 기간: 2026년 8월 24일 ~ 12월 19일

**신청 절차**
1. 생활관 홈페이지 로그인 (KNU 계정)
2. 신청서 작성 (희망 동·룸타입 선택)
3. 합격자 발표: 7월 18일 예정
4. 사생비 납부: 7월 21일 ~ 7월 25일

**선발 기준**
- 거주지(원거리 우선), 성적, 가계 사정 등을 종합 평가합니다.`,
    sources: [
      {
        id: 's-6',
        category: '기숙사',
        title: '2026학년도 2학기 생활관 입사 신청 안내',
        summary: '2026학년도 2학기 생활관 입사 신청은 **6월 16일부터 6월 27일까지** 진행되며, 거주지·성적·가계 사정을 종합하여 선발합니다...',
        publishedAt: '2026-05-19',
        url: 'https://knu.ac.kr/notice/12360',
        highlights: ['6월 16일', '6월 27일', '7월 18일', '거주지', '성적'],
        body: `2026학년도 2학기 생활관 입사 신청 안내`,
      },
    ],
    followups: [
      '사생비는 얼마야?',
      '룸메이트 매칭은 어떻게 돼?',
      '외국인 학생도 신청할 수 있어?',
    ],
    graph: 'dorm',
  },

  // Default fallback
  '__default__': {
    content:
`해당 질문에 대한 답변을 학교 공지사항에서 찾아왔습니다.

질문을 좀 더 구체적으로 입력해주시면 더 정확한 답변을 드릴 수 있어요. 예를 들어:
- 학과·학부를 명시
- 학기·연도를 명시
- 신청·일정·요건 등 키워드 사용`,
    sources: [
      {
        id: 's-default',
        category: '학사',
        title: '경북대학교 학사 일정 안내',
        summary: '2026학년도 학사 일정 전반에 관한 안내사항입니다...',
        publishedAt: '2026-01-05',
        url: 'https://knu.ac.kr/notice/12000',
        highlights: ['학사 일정'],
        body: `경북대학교 학사 일정 전반 안내`,
      },
    ],
    followups: [
      '학사 일정 보여줘',
      '국가장학금 신청은?',
      '졸업 요건 알려줘',
    ],
    graph: 'general',
  },
};

// History items
const HISTORY = [
  {
    id: 'h-current',
    title: '새 대화',
    timeLabel: '방금',
    timeGroup: '오늘',
    isCurrent: true,
  },
  {
    id: 'h-1',
    title: '졸업 요건 관련 질문',
    timeLabel: '오전 11:24',
    timeGroup: '오늘',
  },
  {
    id: 'h-2',
    title: '국가장학금 2학기 신청',
    timeLabel: '어제',
    timeGroup: '어제',
  },
  {
    id: 'h-3',
    title: '복수전공 신청 방법',
    timeLabel: '어제',
    timeGroup: '어제',
  },
  {
    id: 'h-4',
    title: '계절학기 수강신청',
    timeLabel: '5월 15일',
    timeGroup: '이전 7일',
  },
  {
    id: 'h-5',
    title: '학생증 재발급 절차',
    timeLabel: '5월 13일',
    timeGroup: '이전 7일',
  },
];

const BOOKMARKS = [
  {
    id: 'b-1',
    title: '국가장학금 2학기 신청 방법 및 일정',
    snippet: '신청 기간은 5월 27일~6월 26일, 한국장학재단 홈페이지...',
    savedAt: '2일 전',
    category: '장학',
  },
  {
    id: 'b-2',
    title: '컴퓨터학부 졸업 요건 (130학점)',
    snippet: '총 130학점, 전공 72학점, TOEIC 700점 또는 코딩 인증...',
    savedAt: '5일 전',
    category: '학사',
  },
];

// Knowledge graph data
const GRAPH_DATA = {
  courses: {
    title: '수강신청 관련 지식 그래프',
    sub: '관련 공지 4건 · 연결된 개념 6개',
    nodes: [
      { id: 'topic', label: '수강신청', type: 'topic',   x: 0.50, y: 0.45, r: 30 },
      { id: 'd1',    label: '1학기 수강신청 안내', type: 'doc', x: 0.22, y: 0.22, r: 18 },
      { id: 'd2',    label: '수강정정 안내', type: 'doc',     x: 0.78, y: 0.22, r: 16 },
      { id: 'd3',    label: '계절학기 안내', type: 'doc',     x: 0.85, y: 0.55, r: 15 },
      { id: 'd4',    label: '시간표 공지',  type: 'doc',     x: 0.15, y: 0.62, r: 14 },
      { id: 'c1',    label: '학년별 일정', type: 'concept', x: 0.36, y: 0.78, r: 12 },
      { id: 'c2',    label: '포털 접속',  type: 'concept', x: 0.62, y: 0.82, r: 12 },
      { id: 'c3',    label: '폐강 기준', type: 'concept', x: 0.92, y: 0.78, r: 11 },
    ],
    edges: [
      ['topic','d1'], ['topic','d2'], ['topic','d3'], ['topic','d4'],
      ['topic','c1'], ['topic','c2'], ['d2','c3'], ['d1','c1'], ['d1','c2'],
    ],
  },
  scholarship: {
    title: '장학금 관련 지식 그래프',
    sub: '관련 공지 5건 · 연결된 개념 7개',
    nodes: [
      { id: 'topic', label: '장학금',     type: 'topic',   x: 0.50, y: 0.50, r: 32 },
      { id: 'd1',    label: '국가장학금 2학기', type: 'doc', x: 0.20, y: 0.28, r: 18 },
      { id: 'd2',    label: '교내 가계곤란',  type: 'doc', x: 0.80, y: 0.28, r: 16 },
      { id: 'd3',    label: '근로장학생',     type: 'doc', x: 0.20, y: 0.72, r: 15 },
      { id: 'd4',    label: '성적우수 장학', type: 'doc', x: 0.80, y: 0.72, r: 16 },
      { id: 'c1',    label: '가구원 동의', type: 'concept', x: 0.35, y: 0.12, r: 11 },
      { id: 'c2',    label: '서류 제출', type: 'concept', x: 0.65, y: 0.12, r: 11 },
      { id: 'c3',    label: '한국장학재단', type: 'concept', x: 0.12, y: 0.50, r: 12 },
    ],
    edges: [
      ['topic','d1'], ['topic','d2'], ['topic','d3'], ['topic','d4'],
      ['d1','c1'], ['d1','c2'], ['d1','c3'], ['d2','c2'],
    ],
  },
  graduation: {
    title: '졸업 요건 지식 그래프',
    sub: '관련 공지 3건 · 연결된 개념 5개',
    nodes: [
      { id: 'topic', label: '졸업 요건', type: 'topic', x: 0.50, y: 0.50, r: 32 },
      { id: 'd1',    label: '졸업요건 안내', type: 'doc', x: 0.25, y: 0.25, r: 17 },
      { id: 'd2',    label: '졸업 사정', type: 'doc',    x: 0.75, y: 0.25, r: 15 },
      { id: 'c1',    label: '130학점',  type: 'concept', x: 0.20, y: 0.65, r: 14 },
      { id: 'c2',    label: '전공 72학점', type: 'concept', x: 0.50, y: 0.85, r: 13 },
      { id: 'c3',    label: 'TOEIC 700', type: 'concept', x: 0.80, y: 0.65, r: 12 },
      { id: 'c4',    label: '코딩 인증', type: 'concept', x: 0.88, y: 0.40, r: 11 },
    ],
    edges: [
      ['topic','d1'], ['topic','d2'], ['topic','c1'], ['topic','c2'],
      ['topic','c3'], ['topic','c4'], ['d1','c1'], ['d1','c2'],
    ],
  },
  dorm: {
    title: '기숙사 관련 지식 그래프',
    sub: '관련 공지 2건 · 연결된 개념 4개',
    nodes: [
      { id: 'topic', label: '생활관', type: 'topic', x: 0.50, y: 0.50, r: 30 },
      { id: 'd1',    label: '2학기 입사 안내', type: 'doc', x: 0.22, y: 0.28, r: 18 },
      { id: 'd2',    label: '사생비 납부 안내', type: 'doc', x: 0.78, y: 0.28, r: 15 },
      { id: 'c1',    label: '선발 기준',  type: 'concept', x: 0.20, y: 0.72, r: 12 },
      { id: 'c2',    label: '룸타입',    type: 'concept', x: 0.50, y: 0.85, r: 11 },
      { id: 'c3',    label: '입사일',    type: 'concept', x: 0.80, y: 0.72, r: 11 },
    ],
    edges: [
      ['topic','d1'], ['topic','d2'], ['topic','c1'], ['topic','c2'], ['topic','c3'],
      ['d1','c1'], ['d1','c2'], ['d2','c3'],
    ],
  },
  general: {
    title: '관련 지식 그래프',
    sub: '연관 공지 2건',
    nodes: [
      { id: 'topic', label: '학사 일정', type: 'topic', x: 0.50, y: 0.50, r: 28 },
      { id: 'd1', label: '학사 일정 안내', type: 'doc', x: 0.25, y: 0.30, r: 16 },
      { id: 'd2', label: '학사 공지', type: 'doc', x: 0.75, y: 0.30, r: 14 },
    ],
    edges: [['topic','d1'],['topic','d2']],
  },
};

Object.assign(window, {
  SUGGESTED_QUESTIONS,
  CANNED_RESPONSES,
  HISTORY,
  BOOKMARKS,
  GRAPH_DATA,
});
