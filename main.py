import click
import os

@click.command()
def run():
    """Run the Streamlit app."""
    os.system('streamlit run app.py')

if __name__ == '__main__':
    run()