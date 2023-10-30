import openai
import yaml
import argparse

# OpenAI creds
with open('credentials.yaml', 'r') as file:
    # Load the YAML data
    creds = yaml.safe_load(file)



    

        
        
if __name__ == '__main__':
    openai.api_key = creds["OPENAI_API_KEY"]
    openai.organization = creds["OPENAI_ORG_ID"]
    
    parser = argparse.ArgumentParser()
    