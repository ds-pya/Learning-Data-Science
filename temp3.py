{
  "source_topic_distribution": {
    "app": {
      "game":         {"type": "lognorm", "median": 120, "p95": 600},
      "movie":        {"type": "lognorm", "median": 60,  "p95": 300},
      "beauty":       {"type": "lognorm", "median": 45,  "p95": 200},
      "shopping":     {"type": "lognorm", "median": 30,  "p95": 150},
      "stock":        {"type": "lognorm", "median": 20,  "p95": 100},
      "real estate":  {"type": "lognorm", "median": 15,  "p95": 80},
      "politics":     {"type": "lognorm", "median": 20,  "p95": 120},
      "food":         {"type": "lognorm", "median": 30,  "p95": 120},
      "auto and vehicles": {"type": "lognorm", "median": 25, "p95": 100},
      "soccer":       {"type": "lognorm", "median": 25, "p95": 120},
      "travel":       {"type": "lognorm", "median": 35, "p95": 160}
    },
    "you": {
      "game":        {"type": "lognorm", "median": 90,  "p95": 400},
      "celebrity":   {"type": "lognorm", "median": 70,  "p95": 300},
      "beauty":      {"type": "lognorm", "median": 60,  "p95": 250},
      "soccer":      {"type": "lognorm", "median": 40,  "p95": 200},
      "baseball":    {"type": "lognorm", "median": 35,  "p95": 150},
      "travel":      {"type": "lognorm", "median": 50,  "p95": 250},
      "food":        {"type": "lognorm", "median": 45,  "p95": 180},
      "stock":       {"type": "lognorm", "median": 25,  "p95": 120}
    },
    "web": {
      "stock":       {"type": "lognorm", "median": 25,  "p95": 100},
      "politics":    {"type": "lognorm", "median": 25,  "p95": 100},
      "real estate": {"type": "lognorm", "median": 20,  "p95": 80},
      "soccer":      {"type": "lognorm", "median": 20,  "p95": 80},
      "shopping":    {"type": "lognorm", "median": 15,  "p95": 60},
      "food":        {"type": "lognorm", "median": 15,  "p95": 60},
      "beauty":      {"type": "lognorm", "median": 20,  "p95": 80},
      "travel":      {"type": "lognorm", "median": 18,  "p95": 75}
    },
    "ex": {
      "running":     {"type": "poisson", "lambda": 3.0},
      "soccer":      {"type": "poisson", "lambda": 1.0},
      "baseball":    {"type": "poisson", "lambda": 1.0},
      "basketball":  {"type": "poisson", "lambda": 1.0},
      "golf":        {"type": "neg_binom", "mean": 0.5, "var": 1.0},
      "cycle":       {"type": "poisson",  "lambda": 0.8},
      "tennis":      {"type": "poisson",  "lambda": 0.6},
      "volleyball":  {"type": "poisson",  "lambda": 0.5}
    },
    "cal": {
      "travel":      {"type": "zip", "lambda": 0.2, "pi0": 0.7},
      "movie":       {"type": "zip", "lambda": 0.5, "pi0": 0.5},
      "celebrity":   {"type": "zip", "lambda": 0.3, "pi0": 0.6},
      "soccer":      {"type": "zip", "lambda": 0.2, "pi0": 0.7}
    },
    "poi": {
      "travel":      {"type": "zinb", "mean": 0.5, "var": 2.0, "pi0": 0.7},
      "shopping":    {"type": "zinb", "mean": 0.7, "var": 1.5, "pi0": 0.5},
      "camping":     {"type": "zinb", "mean": 0.3, "var": 1.0, "pi0": 0.6}
    },
    "noti": {
      "movie":       {"type": "zip", "lambda": 2.0, "pi0": 0.2},
      "beauty":      {"type": "zip", "lambda": 2.0, "pi0": 0.3},
      "soccer":      {"type": "zip", "lambda": 5.0, "pi0": 0.1},
      "baseball":    {"type": "zip", "lambda": 4.0, "pi0": 0.1}
    }
  }
}