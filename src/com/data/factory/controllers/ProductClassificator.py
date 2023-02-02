from flask import Blueprint
from com.data.factory.services.ProductClassificatorService import ProductClasificatorService


productClassificator = Blueprint('ProductClassificator', __name__)

@productClassificator.route('/product/classificator', methods=['GET'])
def index():
    service = ProductClasificatorService()
    return service.run()
