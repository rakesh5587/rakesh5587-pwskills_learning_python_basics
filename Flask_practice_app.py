from flask import Flask,request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/hello1")  
def hello_world1():
    return "Hello, World!1"


@app.route("/hello2")
def hello_world2():
    return "Hello, World!2"

@app.route("/test_func")
def test():
    a=4+7
    return "this is my first line in the test function {}".format(a)
    
@app.route("/get_input")
def test_input():
    data=request.args.get('x')
    return "this is the input i got from user {}".format(data)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")
