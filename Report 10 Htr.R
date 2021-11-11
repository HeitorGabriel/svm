setwd('/home/heitor/Área de Trabalho/R Projects/Análise Macro/Lab 10')

library(tidyverse)
library(tidymodels)
library(workflowsets)
library(kernlab)
library(ggside)
library(plotly)
library(gridExtra)

dt <- read_csv("letterdata.csv") %>%
	as_tibble()

dt %>% glimpse()

dt <- dt %>% mutate(letter = factor(letter))
dt$letter %>% levels()

dt %>% summary()

dt1 <- dt %>% 
	group_by(letter) %>%
	summarise( Média_pix    = mean(onpix),
			   Var_pix      = sd(onpix)  ,
			   Média_Linha  = mean(xbar),
			   Var_Linha    = sd(xbar)  ,
			   Média_Col    = mean(ybar),
			   Var_Col      = sd(ybar)  )

dt %>% 
	ggplot(	)+
	geom_boxplot(aes(y=onpix,
					 x=letter,
					 color=letter))+
	theme(legend.position = "none")
dt %>% 
	ggplot(	)+
	geom_boxplot(aes(y=xbox,
					 x=letter,
					 color=letter))+
	theme(legend.position = "none")
dt %>% 
	ggplot(	)+
	geom_boxplot(aes(y=ybox,
					 x=letter,
					 color=letter))+
	theme(legend.position = "none")


dt1 %>%
	ggplot() +
	geom_density(aes(Média_pix))
dt1 %>%
	ggplot() +
	geom_density(aes(Média_Linha))
dt1 %>%
	ggplot() +
	geom_density(aes(Média_Col))
			   
slice_1 <- initial_split(dt)
train   <- training(slice_1)
test    <- testing(slice_1)

rbf_svm_algort <- svm_rbf(cost = tune(),
					  rbf_sigma = tune()) %>% 
	set_engine("kernlab") %>% 
	set_mode("classification")

poly_svm_algort <- svm_poly(cost = tune(),
						degree = tune(),
						scale_factor =tune()) %>% 
	set_engine("kernlab") %>% 
	set_mode("classification")

recipe_svm <- 
	recipe(letter ~ .,
		   data = train) %>%
	step_normalize(all_numeric_predictors()) %>% 
	prep()

wrkflw_1 <- workflow() %>%  
	add_model(rbf_svm_algort) %>% 
	add_recipe(recipe_svm)

wrkflw_2 <-   workflow() %>% 
	add_model(poly_svm_algort) %>% 
	add_recipe(recipe_svm)

valid_1 <- vfold_cv(train, v = 5)

start_grid_1 <-
	wrkflw_1 %>% 
	parameters() %>% 
	update(
		cost = cost(c(-1, 1)),
		rbf_sigma = rbf_sigma(c(-2, 2))
	) %>% 
	grid_regular(levels = 1)

start_grid_2 <-
	wrkflw_2 %>% 
	parameters() %>% 
	update(
		cost = cost(c(-1, 1)),
		degree = degree(c(1, 3))
	) %>% 
	grid_regular(levels = 2)

trained_svm_1 <- 
	wrkflw_1 %>% 
	tune_grid(resamples = valid_1,
			  grid = start_grid_1,
			  metrics = metric_set(roc_auc))
trained_svm_2 <- 
	wrkflw_2 %>% 
	tune_grid(resamples = valid_1,
			  grid = start_grid_2,
			  metrics = metric_set(roc_auc))

collect_metrics(trained_svm_1)
collect_metrics(trained_svm_2)

trained_svm_1 %>% show_best(n=15)
trained_svm_2 %>% show_best(n=15)

best_tune  <- select_best(trained_svm_1,
						  'roc_auc',
						  n=1)
final_svm <- rbf_svm_algort %>%
	finalize_model(best_tune)

final_svm

final_svm_wrkflw <- workflow() %>% 
	add_recipe(recipe_svm) %>% 
	add_model(final_svm) %>% 
	last_fit(slice_1) %>% 
	collect_predictions()

final_svm_wrkflw

final_svm_wrkflw %>% count(letter==.pred_class)
