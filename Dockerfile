flowchart TB
    User[User / Conversation]

    subgraph Extract[Extraction Pipeline]
        E1[1. Extract facts]
        E2[2. Semantic dedup]
        E3[3. Categorize]
        E1 --> E2 --> E3
    end

    subgraph L1[Layer 1: PocketMem - SQLite]
        DB[(memory.db)]
        Schema["key, content, category,<br/>importance, access_count,<br/>created_at, accessed_at"]
        Hybrid[Hybrid Retriever<br/>keyword + sqlite-vec]
        DB --- Schema
        DB --> Hybrid
    end

    subgraph Dream[Consolidation Engine - Dreaming]
        Light[Light Phase<br/>scan daily notes]
        Deep[Deep Phase<br/>promote by importance]
        Rem[REM Phase<br/>decay by access freq]
        Light --> Deep --> Rem
    end

    subgraph L2[Layer 2: MEMORY.md]
        MD[MEMORY.md<br/>curated long-term knowledge]
    end

    User --> Extract
    Extract --> DB
    Hybrid -.retrieval.-> User
    DB --> Dream
    Dream --> MD
    MD -.context injection.-> User


flowchart LR
    subgraph Normal[Typical Agent OpenClaw default]
        direction TB
        A1[Conversation]
        A2{Agent judges<br/>worth remembering?}
        A3[Agent writes<br/>directly to MEMORY.md]
        A4[MEMORY.md]
        A1 --> A2
        A2 -->|yes| A3
        A2 -->|no| A1
        A3 --> A4
    end

    subgraph Tini[tiniclaw]
        direction TB
        B1[Conversation]
        B2[Extract to DB<br/>always]
        B3[(PocketMem<br/>SQLite)]
        B4[Consolidation engine<br/>periodic, automatic]
        B5{Importance<br/>+ Access count<br/>+ Recency}
        B6[MEMORY.md]
        B1 --> B2 --> B3
        B3 --> B4 --> B5
        B5 -->|pass threshold| B6
        B5 -->|fail| B3
    end

    style A3 fill:#ffe6e6
    style B4 fill:#e6f7ff