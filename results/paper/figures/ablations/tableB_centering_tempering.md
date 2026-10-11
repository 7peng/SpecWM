**OGBench Scene**

| arm | cube_xyz | arm | drawer | window | buttons | mean |
|---|---|---|---|---|---|---|
| raw kernel | 0.94 | 0.71 | 1.00 | 1.00 | 1.00 | 0.93 |
| + centering | 0.96 | 0.71 | 1.00 | 1.00 | 1.00 | 0.93 |
| + tempering | 0.93 | 0.71 | 0.99 | 0.99 | 0.99 | 0.92 |
| + centering + tempering (paper) | 0.94 | 0.70 | 0.99 | 0.99 | 0.99 | 0.92 |

**CALVIN**

| arm | red_xyz | blue_xyz | pink_xyz | slider | drawer | switch | arm | mean |
|---|---|---|---|---|---|---|---|---|
| raw kernel | 0.79 | 0.85 | 0.80 | 1.00 | 1.00 | 0.99 | 0.55 | 0.85 |
| + centering | 0.85 | 0.89 | 0.87 | 1.00 | 1.00 | 1.00 | 0.59 | 0.88 |
| + tempering | 0.55 | 0.74 | 0.64 | 0.99 | 0.99 | 0.99 | 0.67 | 0.80 |
| + centering + tempering (paper) | 0.76 | 0.82 | 0.79 | 0.99 | 1.00 | 0.99 | 0.66 | 0.86 |
