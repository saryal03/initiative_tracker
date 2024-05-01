# Team 2 Inititative Tracker

from website  import create_app
from flask import Flask

app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)