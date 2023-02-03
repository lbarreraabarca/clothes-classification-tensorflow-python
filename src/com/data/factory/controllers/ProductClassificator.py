from flask import Blueprint, request
from com.data.factory.services.ProductClassificatorService import ProductClasificatorService


productClassificator = Blueprint('ProductClassificator', __name__)

@productClassificator.route('/product/classificator', methods=['POST'])
def index():
    service = ProductClasificatorService()
    return service.run(request.get_json())
