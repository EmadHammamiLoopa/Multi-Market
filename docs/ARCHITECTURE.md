# Architecture — V0 benchmark

At decision index `t`, the predictor receives only bars `<= t`. Future bars are sliced only after prediction. The replay engine validates the returned timestamp and reference price against the current history endpoint.

```text
historical OHLC
      |
      v
point-in-time replay ---> V0 predictor
      |                     |
      |                     v
      |              LONG / SHORT / NO_TRADE
      |                     |
      v                     v
future horizon -------> TP/SL simulator
                            |
                            v
                       costs + result
                            |
                 +----------+----------+
                 |                     |
                 v                     v
              metrics          benchmark history
                 |                     |
                 +----------+----------+
                            v
                      V0/V1/... plots
```

Every benchmark run stores the SHA-256 of the exact source CSV. Version rankings are filtered to the same dataset hash so V1/V2 cannot appear better merely because they were tested on different data.

V0 intentionally uses a simple momentum baseline. Future versions should implement predictors behind the same interface so the historical replay and scoring contract remains comparable.
