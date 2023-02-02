from flask import Blueprint
from com.data.factory.services.Service import Service


controller = Blueprint('ProductClassificator', __name__)

@controller.route('/product/classificator', methods=['POST'])
def index():
    service = Service()
    return service.invoke()
