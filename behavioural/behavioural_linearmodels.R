
# Load packages
library(lmerTest)
library(boot)
library(sjPlot)

# Load data in long format (no R in the docker container so use local path)
data = read.csv('/media/mp/winhdd/2021_npng_final/derivatives/behavioural/behavioural_data.csv')

# Remove too fast and not responded
data = data[(data['rt'] > 200), ]
data = data[data['rt'] < 5000, ]

# Mixed model for acceptance
mod1 = glmer(accept ~ pain_rank*money_rank + (1|sub),
            family='binomial', data=data)
summary(mod1)

# Mixed model for RT with accept
mod3 = lmer(rt ~ pain_rank*money_rank*accept + (1|sub),
           data=data, REML=FALSE)
summary(mod3)

# Mixed model for RT with choice difficulty
mod2 = lmer(rt ~ choicediff + (1|sub),
            data=data, REML=FALSE)
summary(mod2)


