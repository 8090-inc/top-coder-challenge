
## proposed

flowchart TD
    A["🚀 Start: Legacy System<br/>Reverse Engineering"] --> B["📊 Phase 1: Data Exploration"]
    
    B --> B1["Load 1,000 Historical Cases<br/>from public_cases.json"]
    B1 --> B2["Statistical Analysis<br/>& Visualization"]
    B2 --> B3["Pattern Discovery<br/>& Correlation Analysis"]
    B3 --> B4["Initial Hypothesis Formation<br/>Based on Interviews"]
    
    B4 --> C["🔬 Phase 2: Formula Discovery"]
    
    C --> C1["Test Base Per-Diem Theory<br/>($100/day baseline)"]
    C1 --> C2["Test Mileage Tiers Theory<br/>(58¢ first 100mi, then drops)"]
    C2 --> C3["Test Receipt Processing Theory<br/>(caps & diminishing returns)"]
    C3 --> C4["Test Special Cases<br/>(5-day bonus, efficiency)"]
    
    C4 --> D{"Accuracy Check<br/>via eval.sh"}
    D -->|"<90% exact matches"| C1
    D -->|">90% exact matches"| E["⚙️ Phase 3: Implementation"]
    
    E --> E1["Optimize Algorithm<br/>for Performance"]
    E1 --> E2["Create Final run.sh<br/>Script"]
    E2 --> E3["Final Validation<br/>& Testing"]
    
    E3 --> F{"Final Accuracy<br/>Check"}
    F -->|"<95% exact matches"| G["🔧 Debug High-Error<br/>Cases"]
    G --> C1
    F -->|">95% exact matches"| H["✅ Success!<br/>Ready for Submission"]
    
    H --> I["📤 Generate Results<br/>& Submit"]
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style I fill:#c8e6c9
    style D fill:#fff3e0
    style F fill:#fff3e0