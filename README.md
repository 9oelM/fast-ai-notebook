# fast-ai-notebook
Fastai + GCP + Pipenv

1. Pipenv configs
```
pipenv shell

pipenv install 
```

2. Open notebook
```
cd nbs
su # enter your UNIX password. Otherwise you are not going to be able to save files on jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

3. Access notebook at [http://your-external-ip:8888/tree](http://your-external-ip:8888/tree)

## Warning
This is only for your fastai course. NOT for something you might wanna try out privately. This may be **insecure**.
