from flask import Blueprint
from com.data.factory.services.Service import Service


controller = Blueprint('Controller', __name__)

@controller.route('/index', methods=['GET'])
def index():
    service = Service()
    return service.invoke()