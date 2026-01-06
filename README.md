# ID2223-lab2
Lab assignment 2 in ID2223

Authored by Sam Barati and Joel Maharena 

Space: https://huggingface.co/spaces/coolestGuyEver/iris 

## Task 1: 

In this part of the lab, we finetuned an LLM on the Finetome 100k dataset. We closely followed the instructions provided in the sample notebook which covered everything from cleaning the data to finetuning, quantizing, and pushing the model to hugging face. However, the latter part of the process, i.e. conversion to gguf and pushing to HF proved to be a challange. This was eventually resolved by utilizing the llama.cpp library + updating dependencies and splitting the conversion and hub push into separate steps. 

To ensure that we didn't lose any weights due to timeouts on google colab, we checkpointed the weights every 100 steps so that we could resume from where we left off if this were to happen. Doing this was as easy as mounting to drive in our notebook, and setting the 'output_dir' and 'save_steps' arguments in the SFFTrainer. 

The first model we trained, a llama 3.2 1B instruct model turned out rather... dull... if you may. We suspect this to have been due aggressive trainer hyperparameters, e.g. a high learning rate (2e-4) for such a small model, which may have led to catastrophic forgetting: 
<img width="1512" height="859" alt="Screenshot 2025-12-04 at 12 35 40" src="https://github.com/user-attachments/assets/25f6c29b-7229-420d-b0e6-0cc2bff53b16" />

This was remedied by lowering the LR, and increasing warmup steps, resulting in outputs such as: 
<img width="1215" height="854" alt="Screenshot 2025-12-04 at 12 32 04" src="https://github.com/user-attachments/assets/cd0440e1-d341-400e-9611-e7a78cfe8975" />

We'll classify this hyperparameter change in the subsequent section below.

## Task 2

In the last assignment, we realised we had made a model-centric change which improved performance, considering that our model now responded with coherent sentences. Another model-centric change we tested was switching the model to a Phi-3.5-mini which gave slightly better responses, but took much longer to load, which is reasonable considering the size difference (4b params). So with that in mind, we stuck to our smaller model. 

For the data-centric improvement, we sought out to finetune an 'expert' doctor type model that could tutor med-school students. We managed to finetune a llama 1B model on the lavita/ChatDoctor-HealthCareMagic-100k dataset (we used a subset of 5k examples due to compute caps), but this proved to be a bit of a failure since we realised that we had messed up during the data prep stage (you can see this in our notebook) which led to the model responding with the user prompt before answering the question... This means that the "user" role somehow bled into the training set. Due to a lack of compute on Google Colab, we couldn't retrain it, which is a shame. The expected result would be a massive improvement in the models ability to give clear and correct answers to medical questions, which would stand in stark contrast to our 'generalist' finetome model's output. 

Another data-centric approach that we attempted was simply increasing the number of data points in the finetome dataset from 5000 to 10000 (when training the 1B param llama models). However, this wasn't a controlled experiment since we had made the silly mistake of simultaneously changing the learning rate at the same time, meaning that we couldn't confidently claim that the the increase in performance (the increased coherence in the responses as can be see in the screenshots above) is due to the increase in data. Once again, we couldn't re-run the experiments to log the results of increased data due to lack of compute on colab, which is unfortunate, but we believe that an increase in data is going to lead to some improvements due to the fact that the model has more to learn from which can lead to more nuanced and coherent conversations.

RE: Our 'creative' idea for the LLM. We created a tutor of sorts that can give students step by step solutions, explanations and hints. This is done by injecting instructions into the LLM before running inference along with the users question, which essentially then sets the 'mode' of the LLM. Hope you like it. 

Please note that inference can take 30-120 seconds depending on the prompt. Average time should be 60-70 seconds though.
