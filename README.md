# OpenClaw Process-Evolver Plugin — 개발 사양서

> 본 사양서는 시장 조사·컨셉 확정·동작 정의를 마친 결과물.
> 새 세션에서 이 문서로 바로 개발 시작 가능.
> 작성일: 2026-05-12

---

## 1. 프로젝트 정의

### 1.1. 한 줄 정의
> "메모만 남기지 않고, 매뉴얼(SKILL.md)을 *직접 고쳐 쓰고*, 새 도구도 만드는 self-evolving OpenClaw plugin"

### 1.2. 차별화 메시지
> "다른 플러그인들이 *사용자가 말한 것*을 기록할 때, 이 플러그인은 *시스템이 한 일*을 분석한다.
> 그리고 분석 결과로 SKILL.md를 직접 갱신한다."

### 1.3. 시장 포지셔닝
- 기존 1위(pskoett), 3위(ivangdavila)는 **Track 1(사용자 발화 기반)** 중심
- 기존 ExperienceEngine은 **Track 2(행동 기반)지만 프롬프트 주입에만 머무름**
- **본 프로젝트**: Track 2 + **SKILL.md in-place 갱신** (시장 공백)

---

## 2. 핵심 컨셉

### 2.1. Track 분리 원칙
| | Track 1 (Preference) | **Track 2 (Process)** |
|---|---|---|
| 입력 | 사용자 명시적 발화 | tool/skill 실행 로그 |
| 본질 | 사람이 말한 것 | 시스템이 한 일 |
| 트리거 | 발화 감지 | turn/session 종료 |

본 프로젝트는 **Track 2 중심**, Track 1은 가볍게 처리.

### 2.2. 3단 파이프라인
```
[Capture] → [Refine] → [Apply]
```

---

## 3. 동작 사양 (최종 확정)

### 3.1. 입력 3종

| 입력 | 처리 |
|---|---|
| **(1) 사용자 task 교정 발화** | 활성 skill 있으면 자동 등록 / 없으면 후보 제시 |
| **(2) 모든 tool/skill 실행** | 성공·실패 무관 전체 로그 + 관련 SKILL의 "latest execution" 갱신 |
| **(3) 로그 분석 트리거** | 아래 3 분기 |

### 3.2. 로그 분석 트리거 (3분기)

#### (A) 실패 발생
- 실패 즉시 **사유 추정** (네트워크, 권한, rate limit, timeout 등 카테고리화)
- 단순 룰 우선, 애매하면 LLM 보조
- 해당 SKILL.md의 `## Known Failures` 섹션에 즉시 append
- 추정 사유에 따라 **사전 점검(Pre-flight check)** 규칙 자동 생성
- 다음 동일 task 실행 전 사전 점검 수행
- 동일 실패 재발 시: 발생 횟수만 증가

#### (B) 같은 task 3회 이상 실행
- 같은 *task 패턴* 기준 (단순 같은 tool 3회 아님)
- 3회 실행 trace + 기존 SKILL.md 내용을 **LLM에 입력**
- LLM이 개정된 워크플로 제안
- 한번 정제한 패턴은 쿨다운 기간 동안 재호출 금지
- 후보 형식으로 검토 (3.5 참조)

#### (C) 해당 task의 SKILL.md가 없음
- 새 SKILL.md 생성 (LLM 보조)
- 관련 있는 기존 SKILL.md도 보강 (cross-reference 추가)
- INDEX.md에 새 항목 등록
- 후보 형식으로 검토

### 3.3. 검색 방식
1. **1순위**: OpenClaw가 활성화한 `skill_id` 사용
2. **2순위**: tool 이름 + args 임베딩 → 의미 검색
3. **3순위**: 둘 다 매칭 안 되면 → `uncategorized` 영역에 임시 저장

### 3.4. 사용자 교정 발화 처리 — **옵션 ① 채택**
- 활성 skill 컨텍스트에서 발화 → **자동 등록**
- 활성 skill 없을 때 발화 → **후보 형식 제시**
- 자연어 발화는 LLM이 해석해서 적절한 섹션 결정 (Workflow / Preferences / Tools 등)

### 3.5. 후보 검토 시점 — **옵션 ③ (하이브리드) 채택**

| 변경 종류 | 적용 방식 |
|---|---|
| `latest execution` 추가 | **자동** (매 실행마다) |
| 실패 기록 append | **자동** (실패 즉시) |
| 사전 점검 규칙 추가 | **자동** (실패 사유 추정 시) |
| 활성 skill 컨텍스트의 사용자 교정 | **자동** (옵션 ①) |
| 워크플로 재작성 (3회 반복 트리거) | **배치 검토** (후보) |
| 새 skill 생성 (C 트리거) | **배치 검토** (후보) |
| 활성 skill 없을 때 사용자 교정 | **즉시 후보 제시** |

### 3.6. 후보 형식 (표준)
```
[제안된 변경] target: <skill파일경로>
[변경 사유] <트리거 종류 + 근거>
[Diff]
- <삭제될 라인>
+ <추가될 라인>
[승인 / 거절 / 수정] _
```

---

## 4. 저장 구조

### 4.1. 파일 시스템
```
~/.process-evolver/
├── skills/
│   ├── INDEX.md              ← 테이블, 프롬프트 자동 주입
│   ├── <skill-1>.md          ← 개별 skill 매뉴얼
│   ├── <skill-2>.md
│   └── ...
└── db/
    └── evolver.sqlite        ← 로그, 임베딩, 메타데이터
```

### 4.2. INDEX.md 포맷
```markdown
# Skills Index

| Skill | Description | Tags | Last Updated | Path |
|---|---|---|---|---|
| cooking | 음식 조리 워크플로 | food, recipe | 2026-05-10 | skills/cooking.md |
| deployment | 서버 배포 절차 | devops, ci | 2026-05-12 | skills/deployment.md |
```

### 4.3. 개별 SKILL.md 포맷
```markdown
# Skill: <name>

## Description
<한 줄 설명>

## Workflow
<단계별 워크플로>

## Tools Used
<사용 도구 목록>

## Preferences
<사용자 교정 누적>

## Known Failures
- <YYYY-MM-DD> — <tool> 실패 (추정: <사유>)
  - 사유: <상세>
  - 사전 점검: <체크 규칙>
  - 발생: <N>회

## Pre-flight Checks
- <자동 생성된 사전 점검 규칙>

## Latest Executions
| Date | Outcome | Duration | Notes |
|---|---|---|---|
| 2026-05-12 | ✅ | 12분 | ... |
```

### 4.4. SQLite 스키마

```sql
-- tool 호출 단위 raw 로그
tool_calls (
  id, session_id, turn_id,
  skill_id,
  tool_name,
  args_hash, args_json,
  result_summary,
  success bool,
  error_class,        -- network/permission/rate_limit/timeout/invalid_args/dependency_missing/unknown
  error_message,
  failure_reason,     -- LLM 또는 룰 기반 추정 사유
  duration_ms,
  retry_of,           -- 재시도면 직전 call id
  ts
)

-- task 패턴 (같은 task 3회 트리거용)
task_patterns (
  id,
  pattern_hash,       -- task 식별
  skill_id,
  execution_count,
  last_executions_json, -- 최근 N개 trace
  llm_refined_at,     -- 마지막 LLM 정제 시각 (쿨다운용)
  ts
)

-- 후보 (검토 대기)
candidates (
  id,
  type,               -- failure_append / workflow_revise / new_skill / user_correction
  target_path,        -- 대상 SKILL.md
  diff_content,
  reason,
  status,             -- pending / approved / rejected / modified
  created_at,
  reviewed_at
)

-- 적용 이력
applications (
  id,
  candidate_id,
  target_path,
  applied_at,
  outcome_score,      -- 적용 후 효과
  reverted_at
)

-- 사용자 preference (Track 1 가볍게)
preferences (
  id,
  scope,              -- session/skill/global
  skill_id,
  raw_utterance,
  parsed_intent,
  ts
)
```

---

## 5. 기술 스택

### 5.1. 런타임
- **Node.js >= 20**
- **TypeScript** (jiti로 런타임 로드)
- **SQLite** (better-sqlite3)
- **임베딩**: OpenAI/Gemini/Jina API + 로컬 fallback (`@huggingface/transformers`)

### 5.2. OpenClaw 훅 활용
| 훅 | 용도 |
|---|---|
| `before_tool_call` | callId 발급, 시작 시각 기록 |
| `after_tool_call` | 결과·에러·소요시간 기록, 실패 시 사유 추정 트리거 |
| `before_prompt_build` | INDEX.md + 관련 SKILL.md hint 주입 |
| `agent_turn_end` | turn 내 tool 시퀀스 저장 |
| `session_end` | 배치 분석 트리거 (3회 누적 체크, LLM 정제) |
| `user_message_received` | 사용자 발화 분석 (교정 감지) |

---

## 6. 플러그인 구조

```
process-evolver/
├── openclaw.plugin.json         # 매니페스트
├── package.json
├── README.md
├── src/
│   ├── index.ts                 # definePlugin 진입점
│   ├── hooks/
│   │   ├── capture.ts           # before/after_tool_call
│   │   ├── inject.ts            # before_prompt_build
│   │   ├── correction.ts        # user_message_received
│   │   └── finalize.ts          # session_end
│   ├── refine/
│   │   ├── failure-reason.ts    # 실패 사유 추정 (룰 + LLM)
│   │   ├── task-patterns.ts     # 3회 누적 감지
│   │   └── llm-refiner.ts       # LLM 호출해 워크플로 개정
│   ├── apply/
│   │   ├── auto.ts              # 자동 적용 (실패 기록, latest execution)
│   │   ├── candidate.ts         # 후보 생성 및 검토
│   │   ├── skill-md.ts          # SKILL.md 파싱·수정
│   │   └── new-skill.ts         # 새 SKILL.md 생성
│   ├── search/
│   │   ├── active-skill.ts      # 1순위: 활성 skill_id
│   │   └── semantic.ts          # 2순위: 의미 검색
│   ├── storage/
│   │   ├── schema.sql
│   │   ├── db.ts                # SQLite wrapper
│   │   └── skill-files.ts       # SKILL.md 파일 I/O
│   └── cli/
│       └── commands.ts          # /evolve review, /evolve status 등
└── tests/
```

### 6.1. 매니페스트
```json5
{
  "id": "process-evolver",
  "configSchema": {
    "type": "object",
    "properties": {
      "dataPath": { "type": "string", "default": "~/.process-evolver" },
      "autoApplyFailure": { "type": "boolean", "default": true },
      "autoApplyLatestExecution": { "type": "boolean", "default": true },
      "minRecurrenceForRefine": { "type": "integer", "default": 3 },
      "llmCooldownHours": { "type": "integer", "default": 24 },
      "embeddingProvider": { 
        "type": "string", 
        "enum": ["openai", "gemini", "jina", "local"],
        "default": "openai"
      }
    }
  }
}
```

---

## 7. 개발 로드맵

| Phase | 기간 | 산출물 |
|---|---|---|
| **Phase 1: Foundation** | 1주 | 매니페스트, SQLite 스키마, 기본 디렉터리 구조 |
| **Phase 2: Capture** | 1-2주 | before/after_tool_call 훅, raw 로그 수집 |
| **Phase 3: Skill 파일 I/O** | 1주 | INDEX.md + 개별 SKILL.md 읽기/쓰기/파싱 |
| **Phase 4: Auto Apply** | 1-2주 | latest execution 자동 갱신, 실패 즉시 append, 사유 추정 |
| **Phase 5: Pattern Detection** | 1-2주 | task pattern 인식, 3회 누적 감지 |
| **Phase 6: LLM Refiner** | 2주 | 3회 누적 시 LLM 호출, 후보 생성 |
| **Phase 7: Candidate Review** | 1주 | 후보 검토 UI/CLI, diff 표시, 승인 흐름 |
| **Phase 8: New Skill Generation** | 1-2주 | (C) 트리거, 새 SKILL.md 자동 생성 |
| **Phase 9: Search** | 1주 | 활성 skill + 의미 검색 통합 |
| **Phase 10: Polish & Publish** | 1주 | 문서, 데모, ClawHub 등재 |

---

## 8. 핵심 알고리즘 (개발 시 참조)

### 8.1. 실패 사유 추정
```
1. 에러 메시지 키워드 매칭
   - "timeout", "ETIMEDOUT", "ECONNREFUSED" → network
   - "401", "403", "permission denied" → permission
   - "429", "rate limit" → rate_limit
   - "not found", "ENOENT" → dependency_missing
   - "invalid", "validation" → invalid_args

2. 매칭 안 되면 LLM 호출
   - 입력: tool 이름, args, 에러 메시지, 직전 turn 컨텍스트
   - 출력: 사유 카테고리 + 한 줄 설명

3. 사전 점검 규칙 자동 생성 (카테고리별 템플릿)
```

### 8.2. Task 패턴 인식
```
task_hash = hash(
  skill_id || tool_sequence || args_pattern
)

task_sequence_args 정규화:
  - 변수성 높은 값(타임스탬프, 랜덤 ID) → "<VAR>"
  - 의미 단어(파일명, 키워드) → 보존
```

### 8.3. SKILL.md 안전 머지
```
1. SKILL.md 파싱 → 섹션별 AST
2. 새 항목이 들어갈 섹션 결정
3. 기존 항목 중 의미적 유사도 ≥ 0.85 → 강화 (count 증가)
4. 그 외 → append
5. 변경 전후 diff 생성
6. 자동 적용 가능 변경이면 즉시, 아니면 candidates 테이블에 저장
7. git-style history 보존 (롤백 가능)
```

---

## 9. 시장 비교 (재확인)

| Feature | pskoett | ivangdavila | EE | **본 프로젝트** |
|---|---|---|---|---|
| Tool 실행 로그 | △ | ❌ | ✅ | ✅ |
| 실패 사유 추정 | ❌ | ❌ | △ | ✅ |
| **SKILL.md latest execution 갱신** | ❌ | ❌ | ❌ | ✅ ★ |
| **실패 즉시 SKILL.md append** | ❌ | ❌ | ❌ | ✅ ★ |
| **3회 반복 시 LLM 정제** | ❌ | ❌ | △ | ✅ ★ |
| **SKILL.md 없으면 새로 생성 + 주변 보강** | △ | ❌ | ❌ | ✅ ★ |
| INDEX 기반 다단 로딩 | ❌ | △ | ❌ | ✅ ★ |
| 후보 형식 검토 | ❌ | ❌ | △ | ✅ |
| 사용자 발화 자동 등록 | △ | ✅ | ❌ | ✅ |

---

## 10. 참고 자료

- **ClawHub**: https://clawhub.ai/
- **OpenClaw Docs**: https://docs.openclaw.ai/
- **pskoett/self-improving-agent**: https://clawhub.ai/pskoett/self-improving-agent
- **ivangdavila/self-improving**: https://clawhub.ai/ivangdavila/self-improving
- **alan512/ExperienceEngine**: https://clawhub.ai/plugins/@alan512/experienceengine
- **ExperienceEngine 소스**: https://github.com/Alan-512/ExperienceEngine

---

## 11. 다음 세션에서 바로 시작할 작업

1. **Phase 1 시작** — 빈 OpenClaw plugin 보일러플레이트
   - `openclaw.plugin.json` 작성
   - TypeScript 프로젝트 셋업
   - SQLite 초기 스키마

2. **참고 코드 분석**
   - ExperienceEngine 소스의 거버넌스 라이프사이클 구현
   - pskoett의 `scripts/extract-skill.sh` (있다면)

3. **포지셔닝 다듬기**
   - ClawHub 등재용 description
   - README.md draft
