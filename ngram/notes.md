  # 1. Verify pipeline locally (fast, no Modal)
  python scripts/run_local.py --lang lug --scale 100 --classifier naive_bayes                          
                                                                                                       
  # 2. Dry run — inspect job list                                                                      
  modal run modal_app/run_experiments.py --dry-run                                                     
                                                                                                       
  # 3. Full experiment suite (252 parallel Modal containers)                                           
  modal run modal_app/run_experiments.py                                                               
                                                                                                       
  # 4. Aggregate                                                                                     
  python scripts/aggregate.py  

  # 5. Generate LaTeX tables for the paper                                                             
  python scripts/make_tables.py