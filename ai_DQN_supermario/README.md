# README
____

## project name:

    - AI learns and plays SuperMario

## project structure:

    - code: "code" (folder)
    - report: "report.pdf" (file)
    - additional info: "README.md" (file)

## code structure:

    .py:
        - config.py: setup of the parameters
        - model.py: definition of the DQN class
        - replayBuffer.py: definition of the replay experience buffer
        - wrappers.py: environment building
        - train.py: training procedure
        - test.py: testing procedure
        - validation.py: validation of other levels
        - plot_util.py: plotting of the given excel file
        - main.py: main procedure
    .xlsx:
        - data.xlsx: store the data acquiring from the running procedure, which contains episode, training reward and testing reward
    .png:
        - result_pic.png: the plot acquiring from the main procedure
        - result_pic_final.png: plotting of the given excel file
    .dat:
        - SuperMarioBros-1-1-v0.dat: our trained model which we used to generate the results
    floders:
        - models: store the trained model
        - runs: results of every steps of the leatest runing procedure
  

## procedure for code running:

    - environment buildiing:
      - gym (version earlier than 0.25.1)
      - opencv
      - nespy
      - numpy
      - matplotlib
      - pandas
      - pytorch
      - openxyl
      - gc
    - running procedure:
      - adjust the config.py to change the parameters
      - run the command: python main.py
      - get the data and plot
      - acquire a more detailed plot by running the command: python plot_util.py
      - acquire the validation results by adjusting the config parameters and run the command: python validation.py
  
## task aollcation:

    - Ken Chen:
      - project environment building
      - project code programming
      - project code running
      - data and plot processing
      - report "Result" part
    - Pengcheng Ding:
      - project environment building
      - project code programming
      - report "Methodology" part
      - Theoretical analysis
    - Yaohui Chen:
      - project environment building
      - project code programming
      - report "Introduction","Methodlogy" part